import abc

import chains.layers.fully_connected as fc
from chains import env
from chains.graph import node_factory as f
from chains.graph.structure import Node, Graph
from chains.initialization import variable_initializers as init
from chains.operations import regularization_ops as reg
from chains.optimizer import gradient_descent as gd
from chains.tensor.tensor import Dim


class Model(abc.ABC):

    def __init__(self):
        self.cost_graph = None
        self.prediction_graph = None

    # get graphs + choose optimizer + choose monitors
    # metrics: -> Accuracy ? Cost ?
    # hooks: for seeding...
    # BatchNorm: BNLayer() => nouvelles variables. PredictGraph: Fournir mu et sigma
    # => Comment récupérer mu et sigma: Produire mu et sigma dans le graphe ?
    # Predict + que mu et sigma...
    def train(self, x_train, y_train, *, num_iterations, learning_rate, print_cost=False):
        env.seed(3)
        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(self.cost_graph,
                                                learning_rate=learning_rate)
        costs = []
        for i in range(num_iterations):
            env.seed(1)
            optimizer.run()

            if i % 1000 == 0:
                costs.append(optimizer.cost)

            if print_cost and i % 10000 == 0:
                print(f"Cost after iteration {i}: {optimizer.cost}")

        return costs

    def predict(self, x_test):
        self.prediction_graph.placeholders = {self.X: x_test}
        return self.prediction_graph.evaluate()


class Sequence(Model):

    # GraphBuilder actually
    def __init__(self, cnt_features: int, layers, classifier,
                 regularizer=None):  # TODO Input shape + specify m dimension
        super().__init__()
        self.cnt_features = Dim.of(cnt_features)
        self.cnt_examples = Dim.unknown()
        self.X = f.placeholder(shape=(self.cnt_features, self.cnt_examples))  # TODO Allow axis swap
        self.Y = f.placeholder(shape=(1, self.cnt_examples))

        cost_graph, predict_graph, regularizable_vars = self.X, self.X, []
        for pos, layer in enumerate(layers):
            cost_graph, predict_graph, vars = layer.augment(pos, cost_graph, predict_graph)
            regularizable_vars.extend(vars)

        cost_graph, predict_graph = classifier.augment(cost_graph, predict_graph, self.Y)

        if regularizer is not None:
            cost_graph = regularizer.augment(cost_graph, regularizable_vars, self.X)

        self.cost_graph = Graph(cost_graph)
        self.prediction_graph = Graph(predict_graph)


class Layer(abc.ABC):

    def __init__(self):
        self.cost_graph = None
        self.prediction_graph = None
        self.regularizable_vars = []

    @abc.abstractmethod
    def augment(self, pos: int, logits: Node, labels: Node):
        pass


class FullyConnectedLayer(Layer):

    def __init__(self, neurons: int, weight_initializer=None, bias_initializer=None):
        self.neurons = neurons
        self.weight_initializer = init.HeInitializer() if weight_initializer is None else weight_initializer
        self.bias_initializer = init.ZeroInitializer() if bias_initializer is None else bias_initializer

    def augment(self, pos: int, cost_graph: Node, predict_graph: Node):
        cnt_features = cost_graph.shape[0]  # TODO Allow different axis for the "features" and "examples" dimension
        w = f.var("W" + str(pos + 1), self.weight_initializer, shape=(self.neurons, cnt_features))
        b = f.var("b" + str(pos + 1), self.bias_initializer, shape=(self.neurons, 1))
        return fc.fully_connected(cost_graph, w, b, first_layer=(pos == 0)), \
               fc.fully_connected(predict_graph, w, b, first_layer=(pos == 0)), [w]


class ReLuLayer(Layer):

    def augment(self, pos: int, cost_graph: Node, predict_graph: Node):
        return f.relu(cost_graph), f.relu(predict_graph), []


class DropoutLayer(Layer):
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
            return cost_graph + reg.l2_norm_regularizer(self.lambd, f.dim(inputs), vars)
        else:
            return cost_graph
