import os
import string
from math import ceil

import cv2
import keras_ocr
import numpy as np

from node import Node

from .data import preproc as pp
from .data.generator import DataGenerator
from .network.model import HTRModel

class TextClassifier(object):
    """
    Clase para clasificar texto en imágenes.
    """
    def __init__(self):
        """
        Inicializa la instancia con parámetros necesarios.
        """
        # Inicializa el generador de datos y el modelo de reconocimiento de texto
        self.dtgen = DataGenerator(source="iam", batch_size=16, charset=string.printable[:95], max_text_length=128, load_data=False)
        self.model = HTRModel(architecture="puigcerver", input_size=(1024, 128, 1), vocab_size=self.dtgen.tokenizer.vocab_size)
        self.model.load_checkpoint(target="text_model/output/iam/puigcerver/checkpoint_weights.hdf5")
        
        # Inicializa el reconocedor de texto de keras-ocr
        self.recognizer = keras_ocr.recognition.Recognizer(alphabet=string.printable[:36])
        self.recognizer.compile()
        self.pipeline = keras_ocr.pipeline.Pipeline(recognizer=self.recognizer)

    def train_new_data(self):
        """
        Entrena el modelo con nuevos datos.
        """
        self.dtgen.load_data()
        callbacks = self.model.get_callbacks_continue(logdir="text_model/output/iam/puigcerver", checkpoint="text_model/output/iam/puigcerver/checkpoint_weights.hdf5", verbose=1)
        self.model.fit(x=self.dtgen.new_next_train_batch(), epochs=1, steps_per_epoch=self.dtgen.steps['train'],
                       validation_data=self.dtgen.next_valid_batch(), validation_steps=self.dtgen.steps['valid'],
                       callbacks=callbacks, shuffle=True, verbose=1)

    def __set_image(self, image_path):
        """
        Lee la imagen y aplica preprocesamiento.
        """
        image = cv2.imread(image_path, 0)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        ret3, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image

    def __exist_character_x(self, image, x, ymin, ymax):
        """
        Comprueba si existe un carácter en una columna (eje x) dada.
        """
        for y in range(ymin, ymax + 1):
            if image[y, x] == 0:
                return True
        return False

    def __exist_character_y(self, image, y, xmin, xmax):
        """
        Comprueba si existe un carácter en una fila (eje y) dada.
        """
        for x in range(xmin, xmax + 1):
            if image[y, x] == 0:
                return True
        return False

    def __is_collapse(self, A, B):
        """
        Comprueba si dos coordenadas se superponen.
        """
        x1, x2, y1, y2 = A
        x3, x4, y3, y4 = B
        x = max(x1, x3)
        y = max(y1, y3)
        w = min(x2, x4) - x
        h = min(y2, y4) - y
        return w >= 0 and h >= 0

    def __merge_text_nodes(self, image_path, text_coord):
        """
        Fusiona nodos de texto superpuestos.
        """
        text_nodes = text_coord
        image = self.__set_image(image_path)
        EXPAND_MAX = int(image.shape[1] / 20)
        to_delete = []
        for i in text_nodes:
            if i not in to_delete:
                while True:
                    xmin, xmax, ymin, ymax = i
                    collapse_times = 0
                    for j in text_nodes:
                        if j not in to_delete and i != j:
                            if self.__is_collapse([xmin - EXPAND_MAX, xmax + EXPAND_MAX, ymin, ymax], j):
                                xmin_A, xmax_A, ymin_A, ymax_A = i
                                xmin_B, xmax_B, ymin_B, ymax_B = j
                                n_xmin = min(xmin_A, xmin_B)
                                n_xmax = max(xmax_A, xmax_B)
                                n_ymin = min(ymin_A, ymin_B)
                                n_ymax = max(ymax_A, ymax_B)
                                text_nodes[text_nodes.index(i)] = [n_xmin, n_xmax, n_ymin, n_ymax]
                                i = [n_xmin, n_xmax, n_ymin, n_ymax]
                                to_delete.append(j)
                                collapse_times += 1
                    if collapse_times == 0:
                        break
        if len(to_delete) > 0:
            for x in to_delete:
                text_nodes.remove(x)
        EXPAND_MAX = 10
        res = []
        for i in text_nodes:
            xmin, xmax, ymin, ymax = i
            res.append([xmin - EXPAND_MAX, xmax + EXPAND_MAX, ymin - int(EXPAND_MAX / 2), ymax + int(EXPAND_MAX / 2)])
        return res

    def __get_bbox(self, image_path):
        """
        Obtiene las cajas delimitadoras de texto utilizando keras-ocr.
        """
        images = keras_ocr.tools.read(image_path)
        self.image = images
        prediction_groups = self.pipeline.recognize([images])
        texts = []
        results = []
        for ibox in prediction_groups[0]:
            box = ibox[1]
            texts.append(ibox[0])
            xs, ys = set(), set()
            for x in box:
                xs.add(x[0])
                ys.add(x[1])
            results.append(list(map(ceil, [max(ys), min(ys), max(xs), min(xs)])))
        return results, texts

    def draw_boxes(self, image_path, boxes):
        """
        Dibuja las cajas delimitadoras en la imagen.
        """
        image = cv2.imread(image_path)
        for box in boxes:
            xmin, xmax, ymin, ymax = box
            pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], np.int32)
            cv2.polylines(img=image, pts=np.int32([pts]), color=(255, 0, 0), thickness=5, isClosed=True)
        self.__display_image(image, "Text detection")

    def __display_image(self, image, window_name: str):
        """
        Muestra la imagen en una ventana.
        """
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(window_name, cv2.resize(image, (0, 0), fx=0.3, fy=0.3))
        wait_time = 1000
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(wait_time)
            if (keyCode & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break

    def __image_generator(self, image):
        """
        Genera imágenes para la predicción del modelo.
        """
        x_predict = pp.resize_new_data(image, input_size[:2])
        x_predict = np.array([x_predict])
        x_predict = pp.normalization(x_predict)
        yield x_predict

    def recognize(self, image_path):
        """
        Reconoce el texto en la imagen.
        """
        boxes, texts = self.__get_bbox(image_path)
        images = []
        coords = []
        nodes = []
        for box, text in zip(boxes, texts):
            y2, y1, x2, x1 = box
            coords.append([x1, x2, y1, y2])
        coords = self.__merge_text_nodes(image_path, coords)
        self.draw_boxes(image_path, coords)
        for box in coords:
            xmin, xmax, ymin, ymax = box
            crop_img = self.image[ymin:ymax, xmin:xmax]
            images.append(crop_img)
            predict, prob = self.model.predict(x=self.__image_generator(crop_img), steps=1, ctc_decode=True, verbose=1)
            predict = [self.dtgen.tokenizer.decode(x[0]) for x in predict]
            text = predict[0]
            nodes.append(Node(coordinate=[xmin, xmax, ymin, ymax], text=text))
        return zip(nodes, images)
