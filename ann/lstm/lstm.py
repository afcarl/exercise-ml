import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# model = lstm.build_model([1, 50, 100, 1])
def build_model(layers):
    model = Sequential()

    model.add(LSTM(input_shape=(layers[1], layers[0]), return_sequences=True, units=layers[1], activation='relu', kernel_initializer='lecun_uniform'))

    model.add(Dropout(0.2))
    model.add(LSTM(return_sequences=False, units=layers[2], activation='relu', kernel_initializer='lecun_uniform'))

    model.add(Dropout(0.2))
    model.add(Dense(units=layers[3], kernel_initializer='lecun_uniform', activation='relu'))

    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()
    return model


build_model([1, 50, 100, 1])

true_data = np.arange(500)
test_data = true_data.reshape((1,500,1))
predicted_data = true_data
# todo train instead? lol

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(true_data, label='True Data')
plt.plot(predicted_data, label='Prediction')
plt.legend()
plt.show()