# Importa los módulos necesarios
import mnist_loader
import network
import pickle

# Carga los datos de entrenamiento y prueba utilizando el módulo 'mnist_loader'
training_data, test_data, _ = mnist_loader.load_data_wrapper()

# Crea una instancia de la red neuronal con 784 neuronas en la capa de entrada, 30 en la capa oculta y 10 en la capa de salida
net = network.Network([784, 30, 10])

# Entrena la red neuronal utilizando el algoritmo SGD (descenso de gradiente estocástico) durante 30 épocas
# con un tamaño de mini lote de 10 y una tasa de aprendizaje de 0.001, y evalúa su rendimiento en los datos de prueba
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)

# Guarda la red neuronal entrenada en un archivo llamado 'mlRed.pkl' utilizando el módulo 'pickle'
with open('mlRed.pkl', 'wb') as filel:
    pickle.dump(net, filel)

# Sale del programa
exit()

# Aplana una imagen (no se proporciona la implementación de 'aplana') y pasa la imagen a través de la red neuronal
# para obtener un resultado
a = aplana(Imagen)
resultado = net.feedforward(a)
print(resultado)
