import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Dropout, 
                                     Reshape, Bidirectional, LSTM, Dense)
from .layers import FullGatedConv2D, GatedConv2D, OctConv2D

class HTRModel:
    """
    Clase que representa un modelo de reconocimiento de texto a mano.
    """

    def __init__(self, architecture, input_size, vocab_size, greedy=False, beam_width=10, top_paths=1):
        """
        Inicializa un nuevo modelo de reconocimiento de texto a mano.

        Args:
            architecture (str): Nombre de la arquitectura del modelo.
            input_size (tuple): Tamaño de la entrada del modelo.
            vocab_size (int): Tamaño del vocabulario.
            greedy (bool, optional): Si se utiliza una decodificación greedy. Por defecto False.
            beam_width (int, optional): El ancho del beam en la decodificación beam search. Por defecto 10.
            top_paths (int, optional): La cantidad máxima de caminos en la decodificación beam search. Por defecto 1.
        """
        self.architecture = globals()[architecture]
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.model = None
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = max(1, top_paths)

    def summary(self, output=None, target=None):
        """
        Imprime el resumen del modelo en la consola y guarda el resumen en un archivo si se especifica.

        Args:
            output (str, optional): Directorio de salida para guardar el resumen. Por defecto None.
            target (str, optional): Nombre del archivo de destino para guardar el resumen. Por defecto None.
        """
        self.model.summary()

        if target is not None:
            os.makedirs(output, exist_ok=True)
            with open(os.path.join(output, target), "w") as f:
                f.write(self.model.summary())

    def load_checkpoint(self, target):
        """
        Carga los pesos de un modelo desde un archivo de checkpoint.

        Args:
            target (str): Ruta al archivo de checkpoint.
        """
        if os.path.isfile(target):
            if self.model is None:
                self.compile()
            self.model.load_weights(target)

    def get_callbacks_continue(self, logdir, checkpoint, monitor="val_loss", verbose=0):
        """
        Configura los callbacks para continuar el entrenamiento de un modelo.

        Args:
            logdir (str): Directorio de logs para TensorBoard.
            checkpoint (str): Ruta al archivo de checkpoint para guardar los pesos del modelo.
            monitor (str, optional): La métrica a monitorizar. Por defecto "val_loss".
            verbose (int, optional): Nivel de verbosidad. Por defecto 0.

        Returns:
            list: Lista de callbacks configurados.
        """
        callbacks = [
            CSVLogger(filename=os.path.join(logdir, "epochs.log"), separator=";", append=True),
            TensorBoard(log_dir=logdir, histogram_freq=10, profile_batch=0, write_graph=True, write_images=False, update_freq="epoch"),
            ModelCheckpoint(filepath=checkpoint, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=verbose),
            ReduceLROnPlateau(monitor=monitor, min_delta=1e-8, factor=0.2, patience=15, verbose=verbose)
        ]
        return callbacks

    def get_callbacks(self, logdir, checkpoint, monitor="val_loss", verbose=0):
        """
        Configura los callbacks para el entrenamiento de un modelo.

        Args:
            logdir (str): Directorio de logs para TensorBoard.
            checkpoint (str): Ruta al archivo de checkpoint para guardar los pesos del modelo.
            monitor (str, optional): La métrica a monitorizar. Por defecto "val_loss".
            verbose (int, optional): Nivel de verbosidad. Por defecto 0.

        Returns:
            list: Lista de callbacks configurados.
        """
        callbacks = [
            CSVLogger(filename=os.path.join(logdir, "epochs.log"), separator=";", append=True),
            TensorBoard(log_dir=logdir, histogram_freq=10, profile_batch=0, write_graph=True, write_images=False, update_freq="epoch"),
            ModelCheckpoint(filepath=checkpoint, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=verbose),
            EarlyStopping(monitor=monitor, min_delta=1e-8, patience=20, restore_best_weights=True, verbose=verbose),
            ReduceLROnPlateau(monitor=monitor, min_delta=1e-8, factor=0.2, patience=15, verbose=verbose)
        ]
        return callbacks

    def compile(self, learning_rate=0.001):
        """
        Compila el modelo.

        Args:
            learning_rate (float, optional): Tasa de aprendizaje. Por defecto 0.001.
        """
        inputs, outputs = self.architecture(self.input_size, self.vocab_size + 1)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss=self.ctc_loss_lambda_func)

    def fit(self, **kwargs):
        """
        Entrena el modelo.

        Args:
            **kwargs: Argumentos para el método fit de Keras.

        Returns:
            History: Historia del entrenamiento.
        """
        return self.model.fit(**kwargs)

    def predict(self, **kwargs):
        """
        Realiza predicciones utilizando el modelo.

        Args:
            **kwargs: Argumentos para el método predict de Keras.

        Returns:
            ndarray: Predicciones del modelo.
        """
        return self.model.predict(**kwargs)

    @staticmethod
    def ctc_loss_lambda_func(y_true, y_pred):
        """
        Función lambda para calcular la pérdida CTC.

        Args:
            y_true (Tensor): Tensor de valores verdaderos.
            y_pred (Tensor): Tensor de predicciones.

        Returns:
            Tensor: Pérdida CTC.
        """
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        loss = tf.reduce_mean(loss)
        return loss

def puigcerver(input_size, d_model):
    """
    Función que define la arquitectura del modelo según el paper de Puigcerver.

    Args:
        input_size (tuple): Tamaño de la entrada del modelo.
        d_model (int): Dimensión del modelo.

    Returns:
        tuple: Entrada y salida del modelo.
    """
    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(input_data)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

    return (input_data, output_data)
