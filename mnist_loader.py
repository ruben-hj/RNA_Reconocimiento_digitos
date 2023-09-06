import pickle
import gzip
import numpy as np

def load_data():
    """
    Carga los datos del conjunto de datos MNIST almacenados en un archivo comprimido.

    :return: Un tuple que contiene los datos de entrenamiento, validación y prueba.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    Carga y preprocesa los datos del conjunto de datos MNIST para su uso en una red neuronal.

    :return: Un tuple que contiene los datos de entrenamiento, validación y prueba preprocesados.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Convierte una etiqueta de dígito en una representación vectorial.

    :param j: El dígito (0-9) a convertir.
    :return: Un vector que representa el dígito.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
