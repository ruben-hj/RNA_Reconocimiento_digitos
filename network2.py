"""Una versión mejorada de network.py, que implementa el 
algoritmo de aprendizaje de descenso de gradiente estocástico
para una red neuronal de avance. Las mejoras incluyen la 
adición de la función de costo de cross-entropy,
regularización, mejor inicialización de los pesos de la red, 
ademas de la implementacion de optimizador ADAM

"""

import json
import random
import sys
import numpy as np


#### Se Definen las funciones de costo cuadrática y de Cross-entropy

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Devuelve el costo asociado con una salida ``a`` y la salida deseada
         ``y``.

         """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Devuelve el delta de error de la capa de salida."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Devuelve el costo asociado con una salida ``a`` y la salida deseada
         ``y``. Tenga en cuenta que np.nan_to_num se utiliza para garantizar la estabilidad numérica. En particular, si tanto ``a`` como ``y`` tienen un valor 1.0
         en la misma ranura, entonces la expresión (1-y)*np.log(1-a)
         regresa nan. np.nan_to_num garantiza que eso se convierta
         al valor correcto (0,0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Devuelve el delta de error de la capa de salida. Tenga en cuenta que el
         método no utiliza el parámetro ``z``. Esta incluido en
         los parámetros del método para hacer la interfaz
         consistente con el método delta para otras clases de costos.

         """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """La lista ``tamaños`` contiene el número de neuronas en las 
        respectivas capas de la red. Por ejemplo, si la lista fuera [2, 3, 1]
         entonces sería una red de tres capas, con la primera capa
         que contiene 2 neuronas, la segunda capa 3 neuronas y la
         tercera capa 1 neurona. Los sesgos y pesos de la red
         se inicializan aleatoriamente, usando
         ``self.default_weight_initializer`` (ver cadena de documentación para ese
         método).

         """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

        # Parámetros de ADAM
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.m_b = [np.zeros(b.shape) for b in self.biases]
        self.m_w = [np.zeros(w.shape) for w in self.weights]
        self.v_b = [np.zeros(b.shape) for b in self.biases]
        self.v_w = [np.zeros(w.shape) for w in self.weights]

    def default_weight_initializer(self):
        """Inicializa cada peso usando una distribución gaussiana con media 0
         y desviación estándar 1 sobre la raíz cuadrada del número de
         pesos conectados a la misma neurona. Inicializar los sesgos
         usando una distribución gaussiana con media 0 y estándar
         desviación 1.

         Tenga en cuenta que se supone que la primera capa es una capa de entrada, y
         por convención no estableceremos ningún sesgo para esas neuronas, ya que
         Los sesgos sólo se utilizan para calcular los resultados de posteriores
         capas.

         """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Inicializar los pesos usando una distribución gaussiana con media 0
         y desviación estándar 1. Inicialice los sesgos utilizando una
         Distribución gaussiana con media 0 y desviación estándar 1.

         Tenga en cuenta que se supone que la primera capa es una capa de entrada, y
         por convención no estableceremos ningún sesgo para esas neuronas, ya que
         Los sesgos sólo se utilizan para calcular los resultados de las capas 
         posteriores.

         """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Devuelve la salida de la red si se ingresa ``a``."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):
        """Entrene la red neuronal utilizando un descenso de gradiente 
        estocástico de mini lotes. ``training_data`` es una lista de tuplas 
        ``(x, y)`` que representan las entradas de entrenamiento y las 
        salidas deseadas. Los otros parámetros no opcionales se explican por 
        sí mismos, al igual que el parámetro de regularización "lmbda". El 
        método también acepta "datos de evaluación", generalmente datos de 
        validación o de prueba. Podemos monitorear el costo y la precisión de
        los datos de evaluación o de entrenamiento. , configurando los indicadores
        apropiados. El método devuelve una tupla que contiene cuatro listas:
        los costos (por época) de los datos de evaluación, las precisiones de
        los datos de evaluación, los costos de los datos de entrenamiento y las
        precisiones de los datos de entrenamiento. Todos los valores se evalúan 
        al final de cada época de entrenamiento. Entonces, por ejemplo, si
        entrenamos durante 30 épocas, entonces el primer elemento de la tupla
        será una lista de 30 elementos que contiene el costo de los datos de 
        evaluación al final de cada época. Tenga en cuenta que las listas 
        están vacías si no se establece la bandera correspondiente.

         """

         # funcionalidad de parada anticipada:
        best_accuracy=1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # funcionalidad de parada anticipada:
        best_accuracy=0
        no_accuracy_change=0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.t += 1
        learning_rate = eta * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        # Actualización de momentos y pesos con ADAM.
        self.m_b = [(self.beta1 * mb + (1 - self.beta1) * nb) for mb, nb in zip(self.m_b, nabla_b)]
        self.m_w = [(self.beta1 * mw + (1 - self.beta1) * nw) for mw, nw in zip(self.m_w, nabla_w)]
        self.v_b = [(self.beta2 * vb + (1 - self.beta2) * (nb**2)) for vb, nb in zip(self.v_b, nabla_b)]
        self.v_w = [(self.beta2 * vw + (1 - self.beta2) * (nw**2)) for vw, nw in zip(self.v_w, nabla_w)]
        
        self.weights = [(1 - eta * (lmbda / n)) * w - (learning_rate * mb) / (np.sqrt(vb) + self.epsilon)
                        for w, mb, vb in zip(self.weights, self.m_w, self.v_b)]
        self.biases = [b - (learning_rate * nb) / (np.sqrt(vb) + self.epsilon)
                       for b, nb, vb in zip(self.biases, self.m_b, self.v_b)]

    def backprop(self, x, y):
        """Devuelve una tupla ``(nabla_b, nabla_w)`` que representa el
         gradiente para la función de costo C_x. ``nabla_b`` y
         ``nabla_w`` son listas capa por capa de matrices numerosas, similares
         a ``self.biases`` y ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # lista para almacenar todas las activaciones, capa por capa
        zs = [] # lista para almacenar todos los vectores z, capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
      # Tenga en cuenta que la variable l en el bucle siguiente se usa un poco
         # de manera diferente a la notación del Capítulo 2 del libro. Aquí,
         # l = 1 significa la última capa de neuronas, l = 2 es la
         # penúltima capa, y así sucesivamente. Es una renumeración del
         # esquema en el libro, usado aquí para aprovechar el hecho
         # que Python puede usar índices negativos en listas.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Devuelve el número de entradas en ``datos`` para las cuales la 
        red neuronal genera el resultado correcto. Se supone que la salida 
        de la red neuronal es el índice de la neurona de la capa final que 
        tenga la mayor activación.
        La bandera ``convertir`` debe establecerse en Falso si el conjunto 
        de datos son datos de validación o de prueba (el caso habitual), y 
        en Verdadero si el conjunto de datos son datos de entrenamiento. La 
        necesidad de esta bandera surge debido a diferencias en la forma en 
        que se representan los resultados "y" en los diferentes conjuntos de 
        datos. En particular, indica si necesitamos realizar conversiones 
        entre las diferentes representaciones. Puede parecer extraño utilizar 
        diferentes representaciones para los diferentes conjuntos de datos. 
        ¿Por qué no utilizar la misma representación para los tres conjuntos 
        de datos? Se hace por razones de eficiencia: el programa generalmente 
        evalúa el costo de los datos de entrenamiento y la precisión de otros 
        conjuntos de datos. Estos son diferentes tipos de cálculos y el uso de 
        diferentes representaciones acelera las cosas. Se pueden encontrar más 
        detalles sobre las representaciones en mnist_loader.load_data_wrapper.

         """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Carga una red neuronal desde el archivo ``nombre de archivo``. 
    Devuelve una instancia de Red.

     """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Devuelve un vector unitario de 10 dimensiones con un 1,0 en la posición j'
     y ceros en otros lugares. Esto se utiliza para convertir un dígito (0...9)
     en una salida deseada correspondiente de la red neuronal.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """La función sigmoide."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la función sigmoide."""
    return sigmoid(z)*(1-sigmoid(z))
