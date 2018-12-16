import abc

from ..core import node_factory as f, initializers as init
from ..core.graph import Node, Graph
from ..core.static_shape import Dim
from ..utils import validate
from ..utils.naming import NameGenerator


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
        self.layer_names = NameGenerator()
        self.cnt_features = Dim.of(cnt_features)
        self.cnt_samples = Dim.unknown()
        self.inputs = f.placeholder(
            shape=(self.cnt_features, self.cnt_samples))

        cost_graph, predict_graph, regularizable_vars = \
            self.inputs, self.inputs, []
        for pos, layer in enumerate(layers):
            cost_graph, predict_graph, vars = layer.append(pos,
                                                           self.layer_names,
                                                           cost_graph,
                                                           predict_graph)
            regularizable_vars.extend(vars)

        self.cnt_classes = classifier.cnt_classes
        self.label_size = classifier.label_size
        self.labels = f.placeholder(
            shape=(self.label_size, self.cnt_samples))
        cost_graph, predict_graph = classifier.append(self.layer_names,
                                                      cost_graph,
                                                      predict_graph,
                                                      self.labels)

        if regularizer is not None:
            cost_graph = regularizer.append(self.layer_names, cost_graph,
                                            regularizable_vars, self.inputs)

        self.cost_graph = Graph(cost_graph)
        self.predict_graph = Graph(predict_graph)


class SequenceElement(abc.ABC):

    def __init__(self):
        self.layer_name = None
        self.var_name_generator = None

    def new_var_name(self, prefix):
        return self.var_name_generator.generate(prefix)

    def prepare_names(self, name_generator: NameGenerator):
        self.layer_name = name_generator.generate(
            self.__class__.__name__).lower()
        self.var_name_generator = NameGenerator(self.layer_name)


class Layer(SequenceElement, abc.ABC):

    def append(self, pos: int, name_generator: NameGenerator,
               logits: Node, labels: Node):
        self.prepare_names(name_generator)
        return self.do_append(pos, logits, labels)

    @abc.abstractmethod
    def do_append(self, pos: int, logits: Node, labels: Node):
        pass


class Dense(Layer):
    default_weight_initializer = init.HeInitializer()
    default_bias_initializer = init.ZeroInitializer()

    def __init__(self, neurons: int, weight_initializer=None,
                 bias_initializer=None):
        self.neurons = neurons
        self.weight_initializer = self.default_weight_initializer \
            if weight_initializer is None else weight_initializer
        self.bias_initializer = self.default_bias_initializer \
            if bias_initializer is None else bias_initializer

    def do_append(self, pos, cost_g, predict_g):
        # TODO Allow different axis for the "features" and "examples" dimension
        cnt_features = cost_g.shape[0]
        w_name = self.new_var_name("W")
        b_name = self.new_var_name("b")
        w = f.var(w_name, self.weight_initializer,
                  shape=(self.neurons, cnt_features))
        b = f.var(b_name, self.bias_initializer, shape=(self.neurons, 1))
        cost_fc = f.fully_connected(cost_g, w, b, first_layer=(pos == 0),
                                    name=self.layer_name)
        predict_fc = f.fully_connected(predict_g, w, b, first_layer=(pos == 0),
                                       name=self.layer_name + "_p")
        return cost_fc, predict_fc, [w]


class ReLu(Layer):

    def do_append(self, pos, cost_graph, predict_graph):
        return f.relu(cost_graph, name=self.layer_name), \
               f.relu(predict_graph, name=self.layer_name + "_p"), []


class LeakyReLu(Layer):

    def do_append(self, pos, cost_graph, predict_graph):
        return f.leaky_relu(cost_graph, name=self.layer_name), \
               f.leaky_relu(predict_graph, name=self.layer_name + "_p"), \
               []


class Dropout(Layer):
    def __init__(self, keep_prob=0.8):
        if not (0 < keep_prob <= 1):
            raise ValueError(f"Keep probability should be between 0 and 1")
        self.keep_prob = keep_prob

    def do_append(self, pos, cost_graph, predict_graph):
        return f.dropout(self.keep_prob, cost_graph, name=self.layer_name), \
               predict_graph, []


class BatchNorm(Layer):
    default_beta_initializer = init.ZeroInitializer()
    default_gamma_initializer = init.OneInitializer()

    def __init__(self, beta_initializer=None,
                 gamma_initializer=None):
        self.beta_initializer = self.default_beta_initializer \
            if beta_initializer is None else beta_initializer
        self.gamma_initializer = self.default_gamma_initializer \
            if gamma_initializer is None else gamma_initializer

    def do_append(self, pos, cost_graph, predict_graph):
        cnt_features = cost_graph.shape[0]

        beta_name = self.new_var_name("beta")
        gamma_name = self.new_var_name("gamma")
        beta = f.var(beta_name, self.beta_initializer, shape=(cnt_features, 1))
        gamma = f.var(gamma_name, self.gamma_initializer,
                      shape=(cnt_features, 1))

        bnt = f.batch_norm_train(cost_graph, beta, gamma, name=self.layer_name)
        bnp = f.batch_norm_predict(bnt.op, predict_graph, beta, gamma,
                                   name=self.layer_name + "_p")

        return bnt, bnp, [beta, gamma]


class Classifier(SequenceElement, abc.ABC):

    def __init__(self, label_size, cnt_classes):
        self.layer_name = None
        self.label_size = label_size
        self.cnt_classes = cnt_classes

    def append(self, name_generator, cost_graph: Node, predict_graph: Node,
               labels: Node):
        self.prepare_names(name_generator)
        return self.do_append(cost_graph, predict_graph, labels)

    @abc.abstractmethod
    def do_append(self, cost_graph: Node, predict_graph: Node, labels: Node):
        pass


class SigmoidBinaryClassifier(Classifier):

    def __init__(self):
        super().__init__(label_size=1, cnt_classes=2)

    def do_append(self, cost_graph: Node,
                  predict_graph: Node, labels: Node):
        sigmoid = f.sigmoid(predict_graph, name=self.layer_name + "_sigmoid")
        cross_entropy = f.sigmoid_cross_entropy(cost_graph, labels,
                                                name=self.layer_name)
        gt = f.is_greater_than(sigmoid, 0.5, name=self.layer_name + "_gt")

        return cross_entropy, gt


class SoftmaxClassifier(Classifier):  # TODO axis

    def __init__(self, classes: int):
        validate.is_strictly_greater_than("classes", classes, 2)
        super().__init__(label_size=classes, cnt_classes=classes)

    def do_append(self, cost_graph: Node, predict_graph: Node, labels: Node):
        entropy = f.softmax_cross_entropy(cost_graph, labels,
                                          name=self.layer_name)
        softmax = f.softmax(predict_graph, name=self.layer_name + "_softmax")
        argmax = f.argmax(softmax, name=self.layer_name + "_argmax")
        return entropy, argmax


class Regularizer(SequenceElement, abc.ABC):

    def append(self, name_generator, cost_graph: Node, vars, inputs):
        self.prepare_names(name_generator)
        return self.do_append(cost_graph, vars, inputs)

    @abc.abstractmethod
    def do_append(self, cost_graph: Node, vars, inputs):
        pass


class L2Regularizer(Regularizer):

    def __init__(self, lambd=0.8):
        self.lambd = lambd

    def do_append(self, cost_graph: Node, vars, inputs):
        dim = f.dim(inputs, name=self.layer_name + "_dim")
        reg = f.l2_norm_regularizer(self.lambd, dim, vars,
                                    name=self.layer_name)
        if self.lambd > 0:
            return cost_graph + reg
        else:
            return cost_graph
