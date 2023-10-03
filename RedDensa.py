import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import  Adam, RMSprop
from tensorflow.keras.regularizers import L1  # Importamos la regularización L1
import numpy as np

dataset=mnist.load_data() #Cargamos el conjunto de datos MNIST y lo almacenamos en la variable "dataset"

dat=np.array(dataset)
(x_train, y_train), (x_test, y_test) = dataset # Descomponemos el conjunto de datos en datos de entrenamiento y prueba

#Definimos hiperparámetros para el entrenamiento
learning_rate = 0.001
epochs = 25
batch_size = 120

# Definimos hiperparámetros para el entrenamiento
x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
# Convertimos los datos de entrada a tipo float32
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

#Normalizamos los datos
x_trainv /= 255  # x_trainv = x_trainv/255
x_testv /= 255

# Convertimos las etiquetas de clase a codificación one-hot
num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,), kernel_regularizer=L1(l1=1e-5)))  # Capa oculta con regularización L1
model.add(Dense(256, activation='relu', kernel_regularizer=L1(l1=1e-5)))  # otra capa oculta con regularización L1
model.add(Dense(num_classes, activation='softmax'))
#model.summary() #Desglosa la estructura de nuestro modelo

# Compilamos el modelo especificando la función de pérdida, optimizador y métrica de evaluación
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=learning_rate), #cambio el optimizador a RMSprop
              metrics=['accuracy'])

# Entrenamos el modelo en los datos de entrenamiento
history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                   )

test_loss, test_accuracy = model.evaluate(x_testv, y_testc)
print(f"Pérdida en el conjunto de prueba: {test_loss}")
print(f"Precisión en el conjunto de prueba: {test_accuracy}")

#Evaluamos la eficiencia del modelo
'''score = model.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)
a=model.predict(x_testv) #predicción de la red entrenada
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])'''
