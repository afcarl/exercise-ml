from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.datasets import mnist
# use Matplotlib. cause why not
import matplotlib.pyplot as plt

# image dimensions
img_rows, img_cols = 28, 28

# test data, but we wont need everything
(x_train, _), (x_test, _) = mnist.load_data()
# reshape the input dims
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


autoencoder = Sequential()
# convolutional.. cause we are working with images after all
# we apply 32 conv filters with 3x3 size
autoencoder.add(Conv2D(28, (3, 3), activation='relu', border_mode='same', input_shape=input_shape))
autoencoder.add(MaxPooling2D(pool_size=(2,2)))
autoencoder.add(Conv2D(28, (3, 3), activation='relu', border_mode='same'))
autoencoder.add(UpSampling2D(size=(2,2)))
# if the border mode is 'valid',
# the output size will be smaller because the convolution is computed
# only where the input and the filter overlaps
# otherwise, border mode is 'same'
# the output is the same. it will go out of bounds of the input ..
# this area is padded with zeros.
# theano supports 'full'
# but im too noob to understand all the difference, please come back here future
# self and explain this to me!
# to my past self:
# when you are trying to apply the filter on the input, you may choose a
# matrix of lets say 3x3.. but you want to operate with a border,
# meaning you convolve with zeros padded on the rest of the input
# if you are going to that on the edges.. i still dont know why you would want
# to do this though
autoencoder.add(Conv2D(1, (3, 3), activation='relu', border_mode='same'))
autoencoder.compile(loss='mse', optimizer=RMSprop())
autoencoder.summary()

autoencoder.fit(x=x_train, y=x_train, verbose=1, epochs=1, validation_data=(x_test, x_test))

autoencoded_imgs = autoencoder.predict(x_test)
# images to show only show 1/100 of the total
n = 100
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
