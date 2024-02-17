from keras.layers import (Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, TimeDistributed)
from .ROI_pooling_conv import ROIPoolingConv

class CNN(object):
    """
    Clase para construir una Convolutional Neural Network (CNN) para tareas de detección de objetos.
    """

    def __init__(self, num_anchors, rois, num_classes):
        """
        Inicializa la CNN con los parámetros necesarios.
        Args:
            num_anchors (int): Número de anclas utilizadas en la red.
            rois (tuple): Tamaño de las Region of Interest (ROI).
            num_classes (int): Número de clases a predecir.
        """
        super(CNN, self).__init__()

        self.weights_input_path= "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        self.num_anchors = num_anchors
        self.input_rois, self.num_rois = rois
        self.num_classes = num_classes

    @staticmethod
    def get_img_output_length(width, height):
        """Calcula el tamaño de salida de la imagen después de pasar por la red."""
        def get_output_length(input_length):
            return input_length // 16

        return get_output_length(width), get_output_length(height)

    def build_nn_base(self, input_tensor=None):
        """Construye las capas convolucionales base de la red VGG16."""
        input_shape = (None, None, 3)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        bn_axis = 3

        x = self.__get_conv_blocks(img_input)
        return x

    def __get_conv_blocks(self, img_input):
        """Agrega bloques de capas convolucionales a la red."""
        kernel_conv = (3, 3)

        # Bloque 1
        x = Conv2D(
            64,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block1_conv1'
        )(img_input)
        x = Conv2D(
            64,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block1_conv2'
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Bloque 2
        x = Conv2D(
            128,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block2_conv1'
        )(x)
        x = Conv2D(128,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block2_conv2'
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Bloque 3
        x = Conv2D(
            256,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block3_conv1'
        )(x)
        x = Conv2D(
            256,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block3_conv2'
        )(x)
        x = Conv2D(
            256,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block3_conv3'
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Bloque 4
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block4_conv1'
        )(x)
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block4_conv2'
        )(x)
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block4_conv3'
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Bloque 5
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block5_conv1'
        )(x)
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block5_conv2'
        )(x)
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block5_conv3'
        )(x)

        return x

    def create_rpn(self, base_layers):
        """Crea la Red de Proposición de Regiones (RPN) basada en las capas convolucionales de base."""
        x = Conv2D(512,
            (3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='normal',
            name='rpn_conv1'
        )(base_layers)

        x_class = Conv2D(self.num_anchors,
            (1, 1),
            activation='sigmoid',
            kernel_initializer='uniform',
            name='rpn_out_class'
        )(x)
        x_regr = Conv2D(self.num_anchors * 4,
            (1, 1),
            activation='linear',
            kernel_initializer='zero',
            name='rpn_out_regress'
        )(x)

        return [x_class, x_regr, base_layers]

    def build_classifier(self, base_layers, num_classes=21):
        """Construye el clasificador basado en las capas convolucionales de base."""
        pooling_regions = 7
        input_shape = (self.num_rois, 7, 7, 512)

        out_roi_pool = ROIPoolingConv(
                            pooling_regions,
                            self.num_rois
                        )([base_layers, self.input_rois])
        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out_class = TimeDistributed(
                        Dense(
                            num_classes,
                            activation='softmax',
                            kernel_initializer='zero'
                        ),
                        name='dense_class_{}'.format(num_classes)
                    )(out)
        out_regr = TimeDistributed(
                        Dense(
                            4 * (num_classes-1),
                            activation='linear',
                            kernel_initializer='zero'
                        ),
                        name='dense_regress_{}'.format(num_classes)
                    )(out)

        return [out_class, out_regr]
 