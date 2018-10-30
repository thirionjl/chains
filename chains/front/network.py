import abc

import chains.core.node_factory
from chains.core import node_factory as f, initializers as init
from chains.core.graph import Node, Graph
from chains.core.shape import Dim
from chains.utils import validate


class Network(abc.ABC):

    def __init__(self):
        self.cost_graph = None
        self.predict_graph = None
        self.inputs = None
        self.labels = None
        self.label_size = None
        self.cnt_classes = None

    def evaluate(self, x_test):
        self.predict_graph.placeholders = {self.inputs: x_test}
        return self.predict_graph.evaluate()

    def evaluate_cost(self, x_train, y_train):
        self.feed_cost_graph(x_train, y_train)
        return self.cost_graph.evaluate()

    def initialize_variables(self):
        self.cost_graph.initialize_variables()

    def feed_cost_graph(self, x_train, y_train):
        self.cost_graph.placeholders = {self.inputs: x_train,
                                        self.labels: y_train}


class Sequence(Network):

    def __init__(self, cnt_features: int, layers, classifier,
                 regularizer=None):
        super().__init__()
        self.cnt_features = Dim.of(cnt_features)
        self.cnt_samples = Dim.unknown()
        self.inputs = f.placeholder(
            shape=(self.cnt_features, self.cnt_samples))

        cost_graph, predict_graph, regularizable_vars = \
            self.inputs, self.inputs, []
        for pos, layer in enumerate(layers):
            cost_graph, predict_graph, vars = layer.augment(pos,
                                                            self.cnt_samples,
                                                            cost_graph,
                                                            predict_graph)
            regularizable_vars.extend(vars)

        self.cnt_classes = classifier.cnt_classes
        self.label_size = classifier.label_size
        self.labels = f.placeholder(
            shape=(self.label_size, self.cnt_samples))
        cost_graph, predict_graph = classifier.augment(cost_graph,
                                                       predict_graph,
                                                       self.labels)

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
    def augment(self, pos: int, dim_features: Dim, logits: Node, labels: Node):
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

    def augment(self, pos: int, dim_features: Dim, cost_g: Node,
                predict_g: Node):
        # TODO Allow different axis for the "features" and "examples" dimension
        cnt_features = dim_features.value
        w = f.var("W" + str(pos + 1), self.weight_initializer,
                  shape=(self.neurons, cnt_features))
        b = f.var("b" + str(pos + 1), self.bias_initializer,
                  shape=(self.neurons, 1))
        cost_fc = f.fully_connected(cost_g, w, b, first_layer=(pos == 0))
        predict_fc = f.fully_connected(predict_g, w, b, first_layer=(pos == 0))
        return cost_fc, predict_fc, [w]


class ReLu(SequenceElement):

    def augment(self, pos: int, dim_features: Dim, cost_graph: Node,
                predict_graph: Node):
        return f.relu(cost_graph), f.relu(predict_graph), []


class LeakyReLu(SequenceElement):

    def augment(self, pos: int, dim_features: Dim, cost_graph: Node,
                predict_graph: Node):
        return f.leaky_relu(cost_graph), f.leaky_relu(predict_graph), []


class Dropout(SequenceElement):
    def __init__(self, keep_prob=0.8):
        if not (0 < keep_prob <= 1):
            raise ValueError(f"Keep probability should be between 0 and 1")
        self.keep_prob = keep_prob

    def augment(self, pos: int, dim_features: Dim, cost_graph: Node,
                predict_graph: Node):
        return chains.core.node_factory.dropout(self.keep_prob,
                                                cost_graph), predict_graph, []


class BatchNorm(SequenceElement):

    def augment(self, pos: int, dim_features: Dim, cost_graph: Node,
                predict_graph: Node):
        cnt_features = dim_features.value
        beta = f.var("Beta" + str(pos + 1), self.weight_initializer,
                     shape=(cnt_features, 1))
        gamma = f.var("Gamma" + str(pos + 1), self.bias_initializer,
                      shape=(cnt_features, 1))

        bnt = f.batch_norm_train(cost_graph, beta, gamma)
        bnp = f.batch_norm_predict(bnt, predict_graph, beta, gamma)

        return bnt, bnp, [beta, gamma]


class Classifier(abc.ABC):

    def __init__(self, label_size, cnt_classes):
        self.label_size = label_size
        self.cnt_classes = cnt_classes

    @abc.abstractmethod
    def augment(self, cost_graph: Node, predict_graph: Node, labels: Node):
        pass


class SigmoidBinaryClassifier(Classifier):

    def __init__(self):
        super().__init__(label_size=1, cnt_classes=2)

    def augment(self, cost_graph: Node, predict_graph: Node, labels: Node):
        return f.sigmoid_cross_entropy(cost_graph, labels), \
               f.is_greater_than(f.sigmoid(predict_graph), 0.5)


class SoftmaxClassifier(Classifier):  # TODO axis

    def __init__(self, classes: int):
        validate.is_strictly_greater_than("classes", classes, 2)
        super().__init__(label_size=classes, cnt_classes=classes)

    def augment(self, cost_graph: Node, predict_graph: Node, labels: Node):
        return f.softmax_cross_entropy(cost_graph, labels), \
               f.argmax(f.softmax(predict_graph))


class Regularizer(abc.ABC):

    @abc.abstractmethod
    def augment(self, cost_graph: Node, vars, inputs):
        pass


class L2Regularizer(Regularizer):

    def __init__(self, lambd=0.8):
        self.lambd = lambd

    def augment(self, cost_graph: Node, vars, inputs):
        if self.lambd > 0:
            return cost_graph + chains.core.node_factory.l2_norm_regularizer(
                self.lambd,
                f.dim(inputs), vars)
        else:
            return cost_graph
