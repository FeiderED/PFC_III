import sys
from graphviz import Digraph  # Importa la clase Digraph de la biblioteca graphviz
sys.path.append('..')  # Agrega el directorio superior al sys.path
from node import Node  # Importa la clase Node desde un módulo llamado 'node'

class FlowchartGenerator(object):
    def __init__(self, graph, flow, filename):
        super(FlowchartGenerator, self).__init__()
        self.graph_nodes = graph.get_nodes()  # Obtiene los nodos del gráfico
        print(self.graph_nodes)  # Imprime los nodos del gráfico
        self.flow = flow  # Flujo del gráfico
        self.added_nodes = []  # Lista para almacenar nodos ya agregados al diagrama
        self.dot = Digraph(filename="results/"+filename+"flowchart")  # Crea un objeto Digraph
        self.dot.format = 'png'  # Formato de salida del diagrama
        self.DICT = {  # Diccionario para mapear tipos de nodo a formas de diagrama
            'start_end':'ellipse',
            'print':'invhouse',
            'scan':'parallelogram',
            'process':'box',
            'decision':'diamond'
        }

    def generate_flowchart(self):
        # Itera sobre el flujo del gráfico
        for i, key in enumerate(self.flow):
            node = self.flow[key]
            # Si el nodo tiene solo un destino
            if(len(node) == 1):
                cls = self.graph_nodes[key].get_class()  # Obtiene la clase del nodo
                if(self.__is_any_arrow(cls)):  # Si la clase indica una flecha
                    continue  # Salta este nodo
                # Construye el nodo actual
                self.__build_node(cls, self.graph_nodes[key].get_text(), key)
                # Encuentra el siguiente nodo al que conectar
                dest_cls, dest_key = self.__find_dest(node[0])
                if(dest_cls != 'decision'):  # Si no es una decisión
                    self.__build_node(
                        dest_cls,
                        self.graph_nodes[dest_key].get_text(),
                        dest_key
                    )
                    self.__add_edge(str(key), str(dest_key))  # Conecta el nodo actual con el siguiente
                else:
                    if (dest_key in self.added_nodes):  # Si el nodo de destino ya ha sido agregado
                        self.__add_edge(str(key), str(dest_key))  # Conecta el nodo actual con el destino
                    else:
                        self.__add_subgraph(dest_cls, dest_key, last_key=key)  # Agrega un subgrafo
            elif(len(node) == 2):  # Si es una decisión
                arrow_text = self.graph_nodes[node[0]].get_text().lower()
                if(arrow_text == 'si' or arrow_text == 'yes' or arrow_text == 'sí'):  # Si la decisión es 'sí'
                    dest_cls, dest_key = self.__find_dest(self.flow[key][0])
                    if(dest_cls != 'decision'):
                        self.__build_node(
                            dest_cls,
                            self.graph_nodes[dest_key].get_text(),
                            dest_key
                        )
                        self.__add_edge(str(key), str(dest_key), text="Sí")
                    else:
                        self.__add_subgraph(
                            dest_cls,
                            dest_key,
                            last_key=key,
                            text_edge="Sí"
                        )

        self.dot.render(view='false')  # Renderiza el diagrama

    def __is_any_arrow(self, _class):
        return _class.split('_')[0] == "arrow"

    def __build_node(self, _class, text, key):
        if not(key in self.added_nodes):
            type = self.__get_type_node(_class)
            self.dot.node(str(key), label=text, shape=type)
            self.added_nodes.append(key)

    def __add_subgraph(self, _class, key, last_key=None, text_edge=None):
        with self.dot.subgraph() as s:
            s.attr(rank='same')
            text = self.graph_nodes[key].get_text()
            type = self.__get_type_node(_class)
            s.node(str(key), label=text, shape=type)
            if(last_key == None):
                last_key = self.added_nodes[len(self.added_nodes)-1]
            self.added_nodes.append(key)
            self.__add_edge(str(last_key), str(key), text_edge)
            # El rechazo de la condición se selecciona por defecto para estar en el mismo nivel que la forma de decisión.
            arrow_text = self.graph_nodes[self.flow[key][0]].get_text().lower()
            if(arrow_text == 'no'):
                dest_cls, dest_key = self.__find_dest(self.flow[key][0])
            else:
                dest_cls, dest_key = self.__find_dest(self.flow[key][1])

            text = self.graph_nodes[dest_key].get_text()
            type = self.__get_type_node(dest_cls)
            s.node(str(dest_key), label=text, shape=type)
            self.added_nodes.append(dest_key)
            self.__add_edge(str(key), str(dest_key), text='No')


    def __get_type_node(self, _class):
        return self.DICT[_class]

    def __find_dest(self, key):
        node = self.flow[key]
        if(len(node) == 1):
            cls = self.graph_nodes[node[0]].get_class()
            if(self.__is_any_arrow(cls)):
                # recursión
                return self.__find_dest(node[0])
            # encontró destino
            return cls, node[0]
        elif(len(node) == 2):
            pass
        else:
            pass

    def __add_edge(self, origin, dest, text=None):
        if(text == None):
            self.dot.edge(origin, dest)
        else:
            self.dot.edge(origin, dest, label=text)


if __name__ == '__main__':
    # Ejemplo de creación de nodos para el gráfico
    t0 = Node(coordinate=[569,855,110,242], text='inicio')
    t1 = Node(coordinate=[352,1044,482,589], text='a=0, b=0, res=0')
    # Se definen más nodos aquí...
    # Ejemplo de creación de nodos de forma
    s0 = Node(coordinate=[448,1012,57,253], class_shape='start_end')
    # Se definen más nodos de forma aquí...

    # Se crea un objeto Graph con los nodos anteriores
    graph = Graph(
        image_path="",
        text_nodes=[t0, t1],  # Lista de nodos de texto
        shape_nodes=[s0]  # Lista de nodos de forma
    )

    # Se genera el gráfico y se almacena en 'flow'
    flow = graph.generate_graph()
    print(flow)

    # Se crea un objeto FlowchartGenerator con el gráfico y los datos de flujo
    filename = 'sum.dot'
   
