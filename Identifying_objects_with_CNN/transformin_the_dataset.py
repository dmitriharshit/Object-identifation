from keras.datasets import cifar10
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# transforming labels so that they can be used in deep learning algo
# eg 1->[0,1,0,0,0,0,0,0,0,0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
