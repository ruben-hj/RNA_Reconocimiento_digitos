import random
import numpy as np

class Network(object):
    """
    Una implementación simple de una red neuronal de alimentación directa con SGD.
    """

    def __init__(self, sizes):
        """
        Inicializa la red neuronal con tamaños de capas dadas.

        :param sizes: Una lista que contiene el número de neuronas en cada capa.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Calcula la salida de la red neuronal para una entrada dada.

        :param a: Entrada de la red neuronal.
        :return: Salida de la red neuronal.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Entrena la red neuronal utilizando el algoritmo de descenso de gradiente estocástico.

        :param training_data: Datos de entrenamiento como una lista de tuplas (entrada, salida).
        :param epochs: Número de épocas de entrenamiento.
        :param mini_batch_size: Tamaño de los mini lotes para el entrenamiento.
        :param eta: Tasa de aprendizaje.
        :param test_data: Datos de prueba para evaluar la red después de cada época.
        """
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Actualiza los pesos y sesgos de la red neuronal utilizando un mini lote.

        :param mini_batch: Mini lote de datos de entrenamiento.
        :param eta: Tasa de aprendizaje.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Realiza la retropropagación para calcular los gradientes.

        :param x: Entrada de la red.
        :param y: Salida deseada.
        :return: Tuples de gradientes (nabla_b, nabla_w).
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Evalúa la red neuronal en un conjunto de datos de prueba.

        :param test_data: Datos de prueba como una lista de tuplas (entrada, salida).
        :return: Número de entradas clasificadas correctamente.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Calcula la derivada del costo con respecto a la salida de la red.

        :param output_activations: Salida de la red neuronal.
        :param y: Salida deseada.
        :return: Vector de derivadas parciales.
        """
        return (output_activations-y)

def sigmoid(z):
    """
    Calcula la función sigmoide.

    :param z: Valor a calcular la sigmoide.
    :return: Valor de sigmoide.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Calcula la derivada de la función sigmoide.

    :param z: Valor a calcular la derivada de la sigmoide.
    :return: Derivada de sigmoide.
    """
    return sigmoid(z) * (1 - sigmoid(z))
