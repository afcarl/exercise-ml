# playing around with image classification
# on the CIFAR-10 dataset
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
from time import time
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

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
model.add(Conv2D(32, (3, 3), kernel_initializer='lecun_uniform',activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), kernel_initializer='lecun_uniform',activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
# setting dropout probability at 25%
model.add(Dropout(0.25))
# Conv[64] -> Conv[64] -> Pool -> Dropout
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# flatten -> FCC -> Dropout -> Softmax
model.add(Flatten())
model.add(Dense(512, kernel_initializer='lecun_uniform', activation='relu'))
model.add(Dropout(0.50))

model.add(Dense(num_classes, kernel_initializer='lecun_uniform', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['acc'])
model.summary()

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit(x_train, y_train, batch_size=256, epochs=100, verbose=1, validation_data=(x_test, y_test), callbacks=[tensorboard, early_stopping, model_checkpoint])

# TODO - write predictions
prediction = model.predict(x_test)
print(prediction[0])
