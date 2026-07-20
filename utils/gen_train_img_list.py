from torchvision.datasets import ImageFolder
import pickle

dataset = ImageFolder("../Data/MNIST/mnist_train_112")

a = open("../Data/MNIST/mnist.pickle", "wb")
pickle.dump(dataset.imgs, a)
a.close()

# b = open("../../Data/webface21m.pickle", "rb")
# c = pickle.load(b)
# b.close()