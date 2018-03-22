# playing around with image classification
# on the CIFAR-10 dataset
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np

learning_rate = 0.001

# get the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# get the shape (without the number of training data)
input_shape = x_train.shape[1:]

# get the number unique classes
# we only want the shape of the returned uniques
# for our number of classes to help with the softmax later
num_classes = np.unique(y_train).shape[0]

# normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.max(x_train)
x_test /= np.max(x_test)

# one-hot encode the classes
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
# Conv[32] -> Conv[32] -> Pool -> Dropout
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# setting dropout probability at 25%
model.add(Dropout(0.25))
# Conv[64] -> Conv[64] -> Pool -> Dropout
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# flatten -> FCC -> Dropout -> Softmax
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1, validation_data=(x_test, y_test))

# TODO - write predictions
prediction = model.predict(x_test)[0])