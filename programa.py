import mnist_loader
import network
import pickle

training_data, test_data, _ = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)
with open('mlRed.pkl','wb') as filel:
    pickle.dump(net,filel)

exit()
a=aplana(Imagen)
resultado = net.feedforward(a)
print(resultado)