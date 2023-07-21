import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.datasets import cifar10
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# transforming labels so that they can be used in deep learning algo
# eg 1->[0,1,0,0,0,0,0,0,0,0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# first we will normalize the data
X_train = X_train/255.0
X_test = X_test / 255.0

# model:
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu',
          kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu',
          kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# output Layer:
model.add(Dense(10, activation='softmax'))

# training:
opt = SGD(lr=0.05, momentum=0.95)

model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=50, verbose=1, batch_size=64)

# evaluation:
score = model.evaluate(X_test, y_test)
print('accuracy is %s' % score[1])
