from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv2D, Layer, Multiply

class GatedConv2D(Conv2D):
    """
    Capa de convolución con puerta, donde se aplica una función de activación lineal 
    y otra de activación sigmoide a la salida de la convolución, y luego se multiplican 
    ambas salidas elemento a elemento.
    """
    def __init__(self, **kwargs):
        super(GatedConv2D, self).__init__(**kwargs)

    def call(self, inputs):
        # Llama a la función de convolución de la superclase
        output = super(GatedConv2D, self).call(inputs)
        # Aplica la función de activación lineal y sigmoide a la entrada
        linear = Activation("linear")(inputs)
        sigmoid = Activation("sigmoid")(output)
        # Multiplica los resultados de las funciones de activación
        return Multiply()([linear, sigmoid])

    def get_config(self):
        config = super(GatedConv2D, self).get_config()
        return config


class FullGatedConv2D(Conv2D):
    """
    Capa de convolución completamente conectada con puerta, donde se realiza una convolución
    y luego se separa en dos ramas, una para la función de activación lineal y otra para la 
    función de activación sigmoide, y se multiplican ambas salidas elemento a elemento.
    """
    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        # Llama a la función de convolución de la superclase
        output = super(FullGatedConv2D, self).call(inputs)
        # Divide la salida en dos ramas
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])
        # Multiplica los resultados de las ramas
        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config


class OctConv2D(Layer):
    """
    Capa de convolución octavada, donde se realiza una convolución con dos entradas,
    una de alta resolución y otra de baja resolución, y se suman los resultados.
    """
    def __init__(self,
                 filters,
                 alpha,
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding="same",
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        assert alpha >= 0 and alpha <= 1
        assert filters > 0 and isinstance(filters, int)

        super().__init__(**kwargs)

        self.alpha = alpha
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.low_channels = int(self.filters * self.alpha)
        self.high_channels = self.filters - self.low_channels

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
        assert input_shape[0][1] // 2 >= self.kernel_size[0]
        assert input_shape[0][2] // 2 >= self.kernel_size[1]
        assert input_shape[0][1] // input_shape[1][1] == 2
        assert input_shape[0][2] // input_shape[1][2] == 2

        assert K.image_data_format() == "channels_last"
        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel",
                                                   shape=(*self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)
        self.high_to_low_kernel = self.add_weight(name="high_to_low_kernel",
                                                  shape=(*self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_high_kernel = self.add_weight(name="low_to_high_kernel",
                                                  shape=(*self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        self.low_to_low_kernel = self.add_weight(name="low_to_low_kernel",
                                                 shape=(*self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
        super().build(input_shape)

    def call(self, inputs):
        assert len(inputs) == 2
        high_input, low_input = inputs
        # Convoluciones de alta a alta y alta a baja
        high_to_high = K.conv2d(high_input, self.high_to_high_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        high_to_low = K.pool2d(high_input, (2, 2), strides=(2, 2), pool_mode="avg")
        high_to_low = K.conv2d(high_to_low, self.high_to_low_kernel, strides=self.strides,
                               padding=self.padding, data_format="channels_last")

        # Convoluciones de baja a alta y baja a baja
        low_to_high = K.conv2d(low_input, self.low_to_high_kernel,
                               strides=self.strides, padding=self.padding,
                               data_format="channels_last")
        low_to_high = K.repeat_elements(low_to_high, 2, axis=1)
        low_to_high = K.repeat_elements(low_to_high, 2, axis=2)

        low_to_low = K.conv2d(low_input, self.low_to_low_kernel,
                              strides=self.strides, padding=self.padding,
                              data_format="channels_last")

        # Suma de los resultados
        high_add = high_to_high + low_to_high
        low_add = low_to_low + high_to_low

        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)
        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
        }
        return out_config
