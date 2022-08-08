
#import mnist
from conv import Conv33
from keras.datasets import mnist
# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
# train_images = mnist.train_images()
# train_labels = mnist.train_labels()

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(train_X[0].shape)

conv = Conv33(8)
output = conv.forward(train_X[0])
print(output.shape) # (26, 26, 8)