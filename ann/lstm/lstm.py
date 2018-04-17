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


def create_data(sequence_length=50):
    data = np.arange(2000)
    sequence_length = sequence_length + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    print(result)
    normalized_result = []
    # normalize
    for r in result:
        normalized_r = [((float(p) / float(r[0])) - 1) for p in r]
        normalized_result.append(normalized_r)
    
    
    result = np.array(normalized_result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    
   
    return normalised_data

build_model([1, 50, 100, 1])

X_train, y_train, X_test, y_test = create_data()

predicted_data = X_train

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(X_train, label='True Data')
plt.plot(predicted_data, label='Prediction')
plt.legend()
plt.show()