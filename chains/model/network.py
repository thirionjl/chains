import abc

import chains.layers.fully_connected as fc
from chains.graph import node_factory as f
from chains.graph.structure import Node, Graph
from chains.initialization import variable_initializers as init
from chains.operations import regularization_ops as reg
from chains.tensor.tensor import Dim


class Network(abc.ABC):

    def __init__(self):
        self.cost_graph = None
        self.predict_graph = None
        self.inputs = None
        self.labels = None

    def evaluate(self, x_test):
        self.predict_graph.placeholders = {self.inputs: x_test}
        return self.predict_graph.evaluate()

    def evaluate_cost(self, x_train, y_train):
        self.feed(x_train, y_train)
        return self.cost_graph.evaluate()

    def initialize_variables(self):
        self.cost_graph.initialize_variables()

    def feed(self, x_train, y_train):
        self.cost_graph.placeholders = {self.inputs: x_train, self.labels: y_train}


class Sequence(Network):

    # GraphBuilder actually
    def __init__(self, cnt_features: int, layers, classifier, regularizer=None):
        # TODO Input shape + specify m dimension
        super().__init__()
        self.cnt_features = Dim.of(cnt_features)
        self.cnt_examples = Dim.unknown()
        self.inputs = f.placeholder(shape=(
            self.cnt_features, self.cnt_examples))  # TODO Allow axis swap
        self.labels = f.placeholder(shape=(1, self.cnt_examples))  # TODO Allow other shapes

        cost_graph, predict_graph, regularizable_vars = self.inputs, self.inputs, []
        for pos, layer in enumerate(layers):
            cost_graph, predict_graph, vars = layer.augment(pos, cost_graph,
                                                            predict_graph)
            regularizable_vars.extend(vars)

        cost_graph, predict_graph = classifier.augment(cost_graph,
                                                       predict_graph, self.labels)

        if regularizer is not None:
            cost_graph = regularizer.augment(cost_graph, regularizable_vars,
                                             self.inputs)

        self.cost_graph = Graph(cost_graph)
        self.predict_graph = Graph(predict_graph)


class SequenceElement(abc.ABC):

    def __init__(self):
        self.cost_graph = None
        self.prediction_graph = None
        self.regularizable_vars = []

    @abc.abstractmethod
    def augment(self, pos: int, logits: Node, labels: Node):
        pass


class Dense(SequenceElement):
    default_weight_initializer = init.HeInitializer()
    default_bias_initializer = init.ZeroInitializer()

    def __init__(self, neurons: int, weight_initializer=None,
                 bias_initializer=None):
        self.neurons = neurons
        self.weight_initializer = self.default_weight_initializer \
            if weight_initializer is None else weight_initializer
        self.bias_initializer = self.default_bias_initializer \
            if bias_initializer is None else bias_initializer

    def augment(self, pos: int, cost_g: Node, predict_g: Node):
        # TODO Allow different axis for the "features" and "examples" dimension
        cnt_features = cost_g.shape[0]
        w = f.var("W" + str(pos + 1), self.weight_initializer,
                  shape=(self.neurons, cnt_features))
        b = f.var("b" + str(pos + 1), self.bias_initializer,
                  shape=(self.neurons, 1))
        return fc.fully_connected(cost_g, w, b, first_layer=(pos == 0)), \
               fc.fully_connected(predict_g, w, b, first_layer=(pos == 0)), \
               [w]


class ReLu(SequenceElement):

    def augment(self, pos: int, cost_graph: Node, predict_graph: Node):
        return f.relu(cost_graph), f.relu(predict_graph), []


class Dropout(SequenceElement):
    def __init__(self, keep_prob=0.8):
        if not (0 < keep_prob <= 1):
            raise ValueError(f"Keep probability should be between 0 and 1")
        self.keep_prob = keep_prob

    def augment(self, pos: int, cost_graph: Node, predict_graph: Node):
        return reg.dropout(self.keep_prob, cost_graph), predict_graph, []


class Classifier(abc.ABC):

    @abc.abstractmethod
    def augment(self, cost_graph: Node, predict_graph: Node, labels: Node):
        pass


class SigmoidBinaryClassifier(Classifier):

    def augment(self, cost_graph: Node, predict_graph: Node, labels: Node):
        return f.sigmoid_cross_entropy(cost_graph, labels), \
               f.is_greater_than(f.sigmoid(predict_graph), 0.5)


class Regularizer(abc.ABC):

    @abc.abstractmethod
    def augment(self, cost_graph: Node, vars, inputs):
        pass


class L2Regularizer(Regularizer):

    def __init__(self, lambd=0.8):
        self.lambd = lambd

    def augment(self, cost_graph: Node, vars, inputs):
        if self.lambd > 0:
            return cost_graph + reg.l2_norm_regularizer(self.lambd,
                                                        f.dim(inputs), vars)
        else:
            return cost_graph
