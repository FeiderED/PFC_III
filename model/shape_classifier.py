from __future__ import division  # Importación para la compatibilidad con la división en Python 2

import copy
import os
import pickle
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model

# Importaciones locales
from .frcnn.cnn import CNN
from .frcnn.data_generator import Metrics
from .frcnn.roi_helpers import ROIHelpers
from .frcnn.utilities.image_tools import ImageTools

sys.path.append("model")
import frcnn

sys.path.append('..')
from node import Node
from preprocessor import Preprocessor


class ShapeClassifier(object):
    def __init__(
        self,
        results_path,
        bbox_threshold=0.5,
        overlap_thresh_1=0.5,
        overlap_thresh_2=0.3,
        use_gpu=False,
        num_rois=0
    ):
        """
        Inicializa el clasificador de formas.

        Parámetros:
            results_path (str): Ruta donde se encuentran los resultados del modelo entrenado.
            bbox_threshold (float): Umbral para la confianza en la detección de cuadros delimitadores.
            overlap_thresh_1 (float): Umbral de solapamiento para la primera etapa de supresión no máxima.
            overlap_thresh_2 (float): Umbral de solapamiento para la segunda etapa de supresión no máxima.
            use_gpu (bool): Indica si se utilizará la GPU para la inferencia.
            num_rois (int): Número de regiones de interés a considerar.
        """
        super(ShapeClassifier, self).__init__()
        self.results_path = results_path
        self.config = None
        self.__load_config(results_path)
        self.class_mapping = self.config.class_mapping

        if(num_rois > 0):
            self.config.num_rois = num_rois

        self.bbox_threshold = bbox_threshold
        self.overlap_thresh_1 = overlap_thresh_1
        self.overlap_thresh_2 = overlap_thresh_2

        self.class_mapping = {v: k for k, v in self.class_mapping.items()}

        self.colors_class = {
            self.class_mapping[v]:
                np.random.randint(0, 255, 3) for v in self.class_mapping
        }
        if(use_gpu):
            self.__setup()

        self.__build_frcnn()
        self.removable_threshold = 49.0

        self.RECTANGLES_PATH = "../Images/tmp/"

    def __setup(self):
        """
        Configura el uso de la GPU.
        """
        config_gpu = tf.compat.v1.ConfigProto()

        config_gpu.gpu_options.allow_growth = True

        config_gpu.log_device_placement = True
        sess = tf.compat.v1.Session(config=config_gpu)

    def predict_and_save(self, image, image_name, folder_name):
        """
        Realiza predicciones en una imagen y guarda el resultado.

        Parámetros:
            image (numpy.ndarray): La imagen de entrada.
            image_name (str): El nombre de la imagen.
            folder_name (str): El nombre de la carpeta donde se guardará la imagen.
        """
        image = Preprocessor.apply_unsharp_masking(image)

        st = time.time()

        X, ratio = ImageTools.get_format_img_size(image, self.config)
        X = np.transpose(X, (0, 2, 3, 1))

        [Y1, Y2, F] = self.model_rpn.predict(X)

        roi_helper = ROIHelpers(
            self.config,
            overlap_thresh=self.overlap_thresh_1
        )
        R = roi_helper.convert_rpn_to_roi(Y1, Y2)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        bboxes, probs = self.__apply_spatial_pyramid_pooling(R, F)

        img, _ = self.__generate_final_image(
            bboxes,
            probs,
            image,
            roi_helper,
            ratio
        )

        print('Elapsed time = {}'.format(time.time() - st))

        path = self.results_path + "/" + folder_name
        if(os.path.isdir(path) == False):
            os.mkdir(path)

        cv2.imwrite(path + "/" + image_name, img)
        print("Image {}, save in {}".format(image_name, path))

    def predict(self, image, display_image):
        """
        Realiza predicciones en una imagen.

        Parámetros:
            image (numpy.ndarray): La imagen de entrada.
            display_image (bool): Indica si se debe mostrar la imagen con las detecciones.

        Retorna:
            list: Lista de nodos que representan las formas detectadas.
        """
        image = Preprocessor.apply_unsharp_masking(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        X, ratio = ImageTools.get_format_img_size(image, self.config)
        X = np.transpose(X, (0, 2, 3, 1))

        [Y1, Y2, F] = self.model_rpn.predict(X)

        roi_helper = ROIHelpers(
            self.config,
            overlap_thresh=self.overlap_thresh_1
        )
        R = roi_helper.convert_rpn_to_roi(Y1, Y2)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        bboxes, probs = self.__apply_spatial_pyramid_pooling(R, F)

        img, all_dets = self.__generate_final_image(
            bboxes,
            probs,
            image,
            roi_helper,
            ratio
        )
        if(display_image):
            self.__display_image(img, "Shapes detection")

        return self.generate_nodes(all_dets)

    def __display_image(self, image, window_name: str):
        """
        Muestra una imagen en una ventana.

        Parámetros:
            image (numpy.ndarray): La imagen a mostrar.
            window_name (str): El nombre de la ventana.
        """
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(window_name, cv2.resize(image,(0,0),fx=0.5, fy=0.5))

        wait_time = 1000

        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(wait_time)
            if (keyCode & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break

    def __load_config(self, results_path):
        """
        Carga la configuración del modelo desde un archivo.

        Parámetros:
            results_path (str): Ruta donde se encuentran los resultados del modelo entrenado.
        """
        config_path = results_path + "/config.pickle"
        try:
            objects = []
            with (open(config_path, 'rb')) as f_in:
                self.config = pickle.load(f_in)
            print("Config loaded successful!!")
        except Exception as e:
            print("Could not load configuration file, check results path!")
            exit()

    @staticmethod
    def __get_real_coordinates(ratio, x1, y1, x2, y2):
        """
        Calcula las coordenadas reales a partir de las coordenadas normalizadas.

        Parámetros:
            ratio (float): El factor de escala.
            x1 (float): Coordenada x del punto superior izquierdo.
            y1 (float): Coordenada y del punto superior izquierdo.
            x2 (float): Coordenada x del punto inferior derecho.
            y2 (float): Coordenada y del punto inferior derecho.

        Retorna:
            tuple: Coordenadas reales (x1, y1, x2, y2).
        """
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))
        return (real_x1, real_y1, real_x2 ,real_y2)

    def __build_frcnn(self):
        """
        Construye el modelo Faster R-CNN.
        """
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, 512)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.config.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        num_anchors = len(self.config.anchor_box_scales)
        num_anchors *= len(self.config.anchor_box_ratios)

        cnn = CNN(
            num_anchors,
            (roi_input, self.config.num_rois),
            len(self.class_mapping)
        )
        shared_layers = cnn.build_nn_base(img_input)

        rpn_layers = cnn.create_rpn(shared_layers)

        classifier = cnn.build_classifier(
            feature_map_input,
            len(self.class_mapping)
        )

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model(
            [feature_map_input, roi_input],
            classifier
        )
        self.model_classifier = Model([feature_map_input, roi_input], classifier)

        self.__load_weights()

        self.__compile_models()

    def __load_weights(self):
        """
        Carga los pesos preentrenados del modelo.
        """
        model_path = "model/"+self.config.weights_output_path

        try:
            print('Loading weights from {}'.format(model_path))
            self.model_rpn.load_weights(model_path, by_name=True)
            self.model_classifier.load_weights(model_path, by_name=True)
        except Exception as e:
            print('Exception: {}'.format(e))
            print("Couldn't load pretrained model weights!")
            exit()

    def __compile_models(self):
        """
        Compila los modelos RPN y clasificador.
        """
        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')

    def __apply_spatial_pyramid_pooling(self, roi, F):
        """
        Aplica Spatial Pyramid Pooling para generar cajas delimitadoras y probabilidades.

        Parámetros:
            roi (numpy.ndarray): Regiones de interés.
            F (numpy.ndarray): Características extraídas.

        Retorna:
            dict: Cajas delimitadoras.
            dict: Probabilidades.
        """
        bboxes = {}
        probs = {}
        bbox_threshold = self.bbox_threshold
        num_rois = self.config.num_rois

        for jk in range(roi.shape[0] // num_rois + 1):
            ROIs = np.expand_dims(
                roi[num_rois * jk:num_rois * (jk+1), :],
                axis=0
            )
            if ROIs.shape[1] == 0:
                break

            if jk == roi.shape[0] // num_rois:
                # Padding de R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                cond1 = np.max(P_cls[0, ii, :]) < bbox_threshold
                if cond1 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num+1)]
                    tx /= self.config.classifier_regr_std[0]
                    ty /= self.config.classifier_regr_std[1]
                    tw /= self.config.classifier_regr_std[2]
                    th /= self.config.classifier_regr_std[3]
                    x, y, w, h = ROIHelpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass

                stride = self.config.rpn_stride
                bboxes[cls_name].append(
                    [stride * x, stride * y, stride * (x+w), stride * (y+h)]
                )
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        return bboxes, probs

    def __generate_final_image(self, bboxes, probs, img, roi_helper, ratio):
        """
        Genera la imagen final con las detecciones y las coordenadas reales.

        Parámetros:
            bboxes (dict): Cajas delimitadoras.
            probs (dict): Probabilidades.
            img (numpy.ndarray): La imagen original.
            roi_helper (ROIHelpers): Instancia de ROIHelpers.
            ratio (float): Factor de escala.

        Retorna:
            numpy.ndarray: La imagen con las detecciones.
            list: Todas las detecciones con coordenadas reales.
        """
        all_dets = []
        original_img = copy.copy(img)
        i = -1
        for key in bboxes:
            bbox = np.array(bboxes[key])
            roi_helper.set_overlap_thresh(self.overlap_thresh_2)
            new_boxes, new_probs = roi_helper.apply_non_max_suppression_fast(
                bbox,
                np.array(probs[key])
            )

            for jk in range(new_boxes.shape[0]):
                i += 1
                (x1, y1, x2, y2) = new_boxes[jk,:]

                real_coords = self.__get_real_coordinates(ratio, x1, y1, x2, y2)
                all_dets.append((key, 100 * new_probs[jk], real_coords))

        all_dets = self.__fix_detection(all_dets)
        img = self.__draw_rectangles(all_dets, img)

        return img, all_dets

    def __draw_rectangles(self, all_dets, img):
        """
        Dibuja los rectángulos y etiquetas en la imagen.

        Parámetros:
            all_dets (list): Todas las detecciones con coordenadas reales.
            img (numpy.ndarray): La imagen original.

        Retorna:
            numpy.ndarray: La imagen con los rectángulos y etiquetas dibujados.
        """
        for key, prob, coords in all_dets:
            (real_x1, real_y1, real_x2, real_y2) = coords

            cv2.rectangle(
                img,
                (real_x1, real_y1),
                (real_x2, real_y2),
                (
                    int(self.colors_class[key][0]),
                    int(self.colors_class[key][1]),
                    int(self.colors_class[key][2])
                ),
                4
            )

            textLabel = '{}: {}'.format(key, int(prob))
            (retval, baseLine) = cv2.getTextSize(
                textLabel,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                1
            )
            textOrg = (real_x1, real_y1)
            cv2.rectangle(
                img,
                (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
                (22, 166, 184),
                2
            )
            cv2.rectangle(
                img,
                (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
                (255, 255, 255),
                -1
            )
            # Etiqueta de clase
            cv2.putText(
                img,
                textLabel,
                textOrg,
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 0),
                1
            )

        return img

    def __fix_detection(self, all_dets):
        """
        Corrige las detecciones duplicadas o superpuestas.

        Parámetros:
            all_dets (list): Todas las detecciones con coordenadas reales.

        Retorna:
            list: Lista de detecciones corregidas.
        """
        index_to_del = []
        x = 0
        for key, prob, coords in all_dets:
            if x in index_to_del:
                x += 1
                continue

            y = 0
            x1, y1, x2, y2 = coords

            for key_2, prob_2, coords_2 in all_dets:
                if x == y or y in index_to_del:
                    y += 1
                    continue

                flag = False
                offset = False
                no_offset = False
                self.removable_threshold = 30.0

                htal = key_2 == 'arrow_line_left' or key_2 == 'arrow_line_right'
                vtal = key_2 == 'arrow_line_up' or key_2 == 'arrow_line_down'

                # Calcular desplazamiento de coordenadas en detección doble
                if key == key_2:
                    width = x2 - x1
                    height = y2 - y1
                    if key == 'start_end':
                        coords_pos = (x1 + width * 0.2, y1, x2 + width * 0.2, y2)
                        coords_neg = (x1 - width * 0.2, y1, x2 - width * 0.2, y2)
                    elif key == 'print' or key == 'scan' or key == 'process':
                        coords_pos = (x1 + width * 0.05, y1, x2 + width * 0.05, y2)
                        coords_neg = (x1 - width * 0.05, y1, x2 - width * 0.05, y2)
                    elif key == 'arrow_line_right' or key == 'arrow_line_left':
                        coords_pos = (x1, y1 + height * 0.2, x2, y2 + height * 0.2)
                        coords_neg = (x1, y1 - height * 0.2, x2, y2 - height * 0.2)
                    elif key == 'arrow_line_up' or key == 'arrow_line_down':
                        coords_pos = (x1 + width * 0.2, y1, x2 + width * 0.2, y2)
                        coords_neg = (x1 - width * 0.2, y1, x2 - width * 0.2, y2)

                if key == 'print':
                    if key_2 == 'process' or htal or key_2 == 'start_end':
                        no_offset = True
                    elif key_2 == 'print':
                        offset = True
                elif key == 'start_end':
                    if key_2 == 'process' or htal:
                        no_offset = True
                    elif key_2 == 'start_end':
                        offset = True
                elif key == 'scan':
                    if key_2 == 'process':
                        no_offset = True
                    elif key_2 == 'scan':
                        offset = True
                elif key == 'process':
                    if key_2 == 'scan' or htal:
                        no_offset = True
                    elif key_2 == 'process':
                        offset = True
                elif key == 'arrow_line_right':
                    if key_2 == 'arrow_line_left':
                        no_offset = True
                    elif key_2 == 'arrow_line_right':
                        offset = True
                        self.removable_threshold = 15.0
                elif key == 'arrow_line_left':
                    if key_2 == 'arrow_line_right':
                        no_offset = True
                    elif key_2 == 'arrow_line_left':
                        offset = True
                        self.removable_threshold = 15.0
                elif key == 'arrow_line_up':
                    if key_2 == 'arrow_line_down':
                        no_offset = True
                    elif key_2 == 'arrow_line_up':
                        offset = True
                        self.removable_threshold = 12.0
                elif key == 'arrow_line_down':
                    if key_2 == 'arrow_line_up':
                        no_offset = True
                    elif key_2 == 'arrow_line_down':
                        offset = True
                        self.removable_threshold = 12.0
                if no_offset:
                    if self.__is_bbox_removable(coords, coords_2):
                        flag = True
                elif offset:
                    if self.__is_bbox_removable(coords_pos, coords_2):
                        flag = True
                    elif self.__is_bbox_removable(coords_neg, coords_2):
                        flag = True

                if flag:
                    if prob < prob_2:
                        index_to_del.append(x)
                        break
                    else:
                        index_to_del.append(y)
                y += 1
            x += 1

        _all_dets = []
        x = 0
        for key, prob, coords in all_dets:
            if x in index_to_del:
                x += 1
                continue
            _all_dets.append((key, prob, coords))
            x += 1

        return _all_dets

    def __is_bbox_removable(self, coords, coords_2):
        """
        Verifica si una caja delimitadora es removible debido a la superposición.

        Parámetros:
            coords (tuple): Coordenadas de la primera caja delimitadora.
            coords_2 (tuple): Coordenadas de la segunda caja delimitadora.

        Retorna:
            bool: True si la caja delimitadora es removible, False en caso contrario.
        """
        inter_area = Metrics.intersection(coords, coords_2)
        x1, y1, x2, y2 = coords_2
        area_2 = float((x2 - x1) * (y2 - y1))
        percent_area = float((inter_area * 100)) / float(area_2)

        return percent_area > self.removable_threshold

    def generate_nodes(self, dets):
        """
        Genera los nodos correspondientes a las detecciones.

        Parámetros:
            dets (list): Lista de detecciones.

        Retorna:
            list: Lista de nodos.
        """
        nodes = []
        for det in dets:
            x1, y1, x2, y2 = det[2]
            cord = [x1, x2, y1, y2]
            _node = Node(
                str(det[0]) + '_' + str(det[1]),
                str(det[0]),
                cord,
                det[1]
            )
            nodes.append(_node)

        return nodes

    def generate_edges(self, nodes):
        """
        Genera los bordes correspondientes a los nodos.

        Parámetros:
            nodes (list): Lista de nodos.

        Retorna:
            list: Lista de bordes.
        """
        edges = []
        for node_1 in nodes:
            for node_2 in nodes:
                if node_1.id != node_2.id:
                    weight = Metrics.euclidean_distance(node_1.cord, node_2.cord)
                    _edge = Edge(node_1.id, node_2.id, weight)
                    edges.append(_edge)

        return edges

    def draw_graph(self, nodes, edges):
        """
        Dibuja el grafo correspondiente a los nodos y bordes.

        Parámetros:
            nodes (list): Lista de nodos.
            edges (list): Lista de bordes.
        """
        G = nx.Graph()
        for node in nodes:
            G.add_node(node.id)
        for edge in edges:
            G.add_edge(edge.src, edge.dst, weight=edge.weight)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()



    def generate_model(self, img):
        """
        Genera el modelo con las detecciones.

        Parámetros:
            img (numpy.ndarray): La imagen original.

        Retorna:
            numpy.ndarray: La imagen con las detecciones.
        """
        ratio = 1
        X, ratio = self.__format_img(img, self.config)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # obtiene la característica de salida del modelo rpn [P_cls, P_regr]
        [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(
            Y1,
            Y2,
            self.config,
            K.image_dim_ordering(),
            overlap_thresh=self.config.rpn_overlap_max,
            max_boxes=300
        )

        # convierte R (x_roi) en un conjunto de formato (x1, y1, x2, y2)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # clasificación por box y regrasión por box
        bboxes, probs = self.__apply_spatial_pyramid_pooling(R, F)

        # superpone el bbox en la imagen y muestra
        img, all_dets = self.__generate_final_image(
            bboxes,
            probs,
            img,
            roi_helpers,
            ratio
        )

        # crea el grafo
        nodes = self.generate_nodes(all_dets)
        edges = self.generate_edges(nodes)

        # dibuja el grafo
        self.draw_graph(nodes, edges)

        return img
