import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy

class LossesCalculator(object):

    # Atributos estáticos
    lambda_rpn_regr = 1.0
    lambda_rpn_class = 1.0
    lambda_cls_regr = 1.0
    lambda_cls_class = 1.0
    epsilon = None
    num_classes = None
    num_anchors = None

    def __init__(self, num_classes, num_anchors, epsilon=1e-4):
        super(LossesCalculator, self).__init__()

        # Inicialización de los atributos estáticos
        LossesCalculator.epsilon = epsilon
        LossesCalculator.num_classes = num_classes
        LossesCalculator.num_anchors = num_anchors

    @staticmethod
    def rpn_loss_regr():
        """Calcula la pérdida para la regresión en RPN."""

        def rpn_loss_regr_fixed_num(y_true, y_pred):
            # Obtiene el número de anclajes por capa
            anchors = LossesCalculator.num_anchors
            # Calcula la diferencia entre la verdad y la predicción
            x = y_true[:, :, :, 4 * anchors:] - y_pred
            # Obtiene el valor absoluto de la diferencia
            x_abs = K.abs(x)
            # Convierte a booleano si la diferencia es menor o igual a 1
            x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

            # Obtiene los valores de lambda y epsilon
            lbda_rpn_regr = LossesCalculator.lambda_rpn_regr
            eps = LossesCalculator.epsilon
            # Selecciona las etiquetas verdaderas
            y_sel = y_true[:, :, :, :4 * anchors]
            # Calcula la suma de los términos de la pérdida
            sum1 = K.sum(y_sel * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)))
            sum2 = K.sum(eps + y_true[:, :, :, :4 * anchors])
            # Devuelve la pérdida normalizada
            return lbda_rpn_regr * sum1 / sum2

        return rpn_loss_regr_fixed_num

    @staticmethod
    def rpn_loss_cls():
        """Calcula la pérdida para la clasificación en RPN."""

        def rpn_loss_cls_fixed_num(y_true, y_pred):
            # Obtiene el valor de lambda y epsilon
            lbda_rpn_cls = LossesCalculator.lambda_rpn_class
            anchors = LossesCalculator.num_anchors
            eps = LossesCalculator.epsilon
            # Calcula la pérdida de entropía cruzada binaria
            binary_crossentropy = K.binary_crossentropy(
                y_pred[:, :, :, :],
                y_true[:, :, :, anchors:]
            )
            # Calcula la suma ponderada de la pérdida
            sum1 = K.sum(y_true[:, :, :, :anchors] * binary_crossentropy)
            sum2 = K.sum(eps + y_true[:, :, :, :anchors])
            # Devuelve la pérdida normalizada
            return lbda_rpn_cls * sum1 / sum2

        return rpn_loss_cls_fixed_num

    @staticmethod
    def class_loss_regr():
        """Calcula la pérdida para la regresión en la clasificación."""

        def class_loss_regr_fixed_num(y_true, y_pred):
            # Obtiene el número de clases
            num_classes = LossesCalculator.num_classes - 1
            # Calcula la diferencia entre la verdad y la predicción
            x = y_true[:, :, 4 * num_classes:] - y_pred
            # Obtiene el valor absoluto de la diferencia
            x_abs = K.abs(x)
            # Convierte a booleano si la diferencia es menor o igual a 1
            x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

            # Obtiene los valores de lambda y epsilon
            lbda_cls_regr = LossesCalculator.lambda_cls_regr
            eps = LossesCalculator.epsilon
            # Selecciona las etiquetas verdaderas
            y_sel = y_true[:, :, :4 * num_classes]
            # Calcula la suma ponderada de los términos de la pérdida
            sum1 = K.sum(y_sel * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)))
            sum2 = K.sum(eps + y_true[:, :, :4 * num_classes])
            # Devuelve la pérdida normalizada
            return lbda_cls_regr * sum1 / sum2

        return class_loss_regr_fixed_num

    @staticmethod
    def class_loss_cls(y_true, y_pred):
        """Calcula la pérdida para la clasificación en la clasificación."""

        # Obtiene el valor de lambda
        lbda_cls_class = LossesCalculator.lambda_cls_class
        # Calcula la media de la entropía cruzada categórica
        mean = K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
        # Devuelve la pérdida ponderada
        return lbda_cls_class * mean
