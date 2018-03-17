from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
# use Matplotlib. cause why not
import matplotlib.pyplot as plt

# total pixels of flattened mnist
input_dim = 784
# setting learning rate
learning_rate = 0.001

autoencoder = Sequential()
# lecun uniform with relus 
autoencoder.add(Dense(30, kernel_initializer='lecun_uniform', input_dim=input_dim, activation='relu'))
autoencoder.add(Dense(30, kernel_initializer='lecun_uniform', activation='relu'))
# creating a bottleneck to reduce dimensionality
autoencoder.add(Dense(10, kernel_initializer='lecun_uniform', activation='relu'))
# reconstruction layer
autoencoder.add(Dense(30, kernel_initializer='lecun_uniform', activation='relu'))
autoencoder.add(Dense(30, kernel_initializer='lecun_uniform', activation='relu'))
autoencoder.add(Dense(input_dim, activation='linear'))
autoencoder.compile(loss='mse', optimizer=Adam(lr=learning_rate))
autoencoder.summary()

# preparing test data

# test data, but we wont need everything
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshapre the input dims
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

autoencoder.fit(x=x_train, y=x_train, verbose=1, epochs=1, validation_data=(x_test, x_test))

autoencoded_imgs = autoencoder.predict(x_test)
# images to show only show 1/100 of the total
n = 20
fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(wspace=0.2, top=0.25)

for i in range(1, n + 1):
    encoded = fig.add_subplot(1, n, i)
    encoded.get_xaxis().set_visible(False)
    encoded.get_yaxis().set_visible(False)
    plt.imshow(autoencoded_imgs[i-1].reshape(28, 28), cmap='gray')
    
    original = fig.add_subplot(2, n, i)
    original.get_xaxis().set_visible(False)
    original.get_yaxis().set_visible(False)
    plt.imshow(x_test[i-1].reshape(28, 28), cmap='gray')        
    
plt.show()


