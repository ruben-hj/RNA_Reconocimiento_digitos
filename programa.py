# Importamos los módulos necesarios
import mnist_loader  # Módulo para cargar datos MNIST
import network2      # Nuestra implementación de la red neuronal

# Cargamos los datos de entrenamiento, validación y prueba desde el módulo mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Convertimos los datos de entrenamiento en una lista (para reutilizarlos)
training_data = list(training_data)

# Creamos una instancia de la red neuronal
# Parámetros:
# - [784, 50, 10]: Tamaños de las capas de la red. 784 neuronas de entrada, 30 en la capa oculta y 10 en la capa de salida.
# - cost=network2.CrossEntropyCost: Usaremos la función de costo de entropía cruzada como criterio de entrenamiento.
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

# Inicializamos los pesos de la red neuronal con la inicialización "large_weight_initializer"
net.large_weight_initializer()

# Entrenamos la red neuronal utilizando el algoritmo de SGD (Stochastic Gradient Descent)
# Parámetros:
# - training_data: Datos de entrenamiento.
# - 50: Número de épocas de entrenamiento.
# - 10: Tamaño del mini lote (mini-batch size).
# - 0.1: Tasa de aprendizaje (learning rate).
# - lmbda=5.0: Parámetro de regularización (lambda).
# - evaluation_data=validation_data: Datos de validación para monitorear el rendimiento durante el entrenamiento.
# - monitor_evaluation_accuracy=True: Habilita el monitoreo de la precisión en los datos de validación.
net.SGD(training_data, 50, 10, 0.1, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)
