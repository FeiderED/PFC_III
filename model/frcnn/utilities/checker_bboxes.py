import random
from parser import Parser  # Importa la clase Parser desde un módulo llamado 'parser'

import cv2
from image_tools import ImageTools  # Importa la clase ImageTools desde un módulo llamado 'image_tools'


class Checker(object):
    """
    Clase para verificar y mostrar muestras de imágenes con anotaciones.
    """

    def __init__(self, dataset_path, annotate_path):
        """
        Inicializa la clase Checker con las rutas del conjunto de datos y las anotaciones.

        Args:
            dataset_path (str): Ruta del conjunto de datos.
            annotate_path (str): Ruta del archivo de anotaciones.
        """
        super(Checker, self).__init__()
        self.dataset_path = dataset_path
        self.annotate_path = annotate_path
        self.all_imgs = []
        self.__load_data()

    def __load_data(self):
        """
        Carga los datos de las imágenes y las anotaciones utilizando el Parser.
        """
        parser = Parser(
            dataset_path=self.dataset_path,
            annotate_path=self.annotate_path,
        )
        # Obtiene los datos sin generar anotaciones
        self.all_imgs, _, _ = parser.get_data(generate_annotate=False)

    def show_samples(self, num_imgs):
        """
        Muestra muestras de imágenes con anotaciones.

        Args:
            num_imgs (int): Número de imágenes a mostrar.
        """
        train_images = [s for s in self.all_imgs if s['imageset'] == 'trainval']
        num_imgs = num_imgs if num_imgs < len(train_images) else len(train_images)
        random.shuffle(train_images)
        some_imgs = train_images[:num_imgs]
        print("." * 45)

        for img in some_imgs:
            path = img['filepath']
            image = cv2.imread(path)
            width = img['width']
            height = img['height']
            bboxes = img['bboxes']

            for bbox in bboxes:
                _class = bbox['class']
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']

                # Dibuja el rectángulo de la región de interés (ROI)
                cv2.rectangle(image, (x1, y1), (x2, y2), (170, 30, 5), 3)

                # Agrega un cuadro para el texto de la clase
                (retval, baseLine) = cv2.getTextSize(
                    _class,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    1
                )
                textOrg = (x1, y1)

                cv2.rectangle(
                    image,
                    (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                    (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
                    (22, 166, 184),
                    2
                )

                cv2.rectangle(
                    image,
                    (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                    (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
                    (255, 255, 255),
                    -1
                )

                # Agrega el texto de la clase
                cv2.putText(
                    image,
                    _class,
                    textOrg,
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 0),
                    1
                )

            # Ajusta el tamaño de la imagen
            (width, height) = ImageTools.get_new_img_size(width, height)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
            # Muestra la imagen
            cv2.imshow('sample', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    dataset_path = "/flowchart"
    annotate_path = "/model/frcnn/utilities/annotate.txt"
    checker = Checker(
        dataset_path=dataset_path,
        annotate_path=annotate_path
    )
    checker.show_samples(80)  # Muestra 80 muestras de imágenes con anotaciones
