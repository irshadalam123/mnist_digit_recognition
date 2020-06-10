from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255
x_test = x_test/255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)

def createModel():
    model = Sequential()

    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation = "softmax"))

    model.compile(optimizer = "adam",
                loss = "categorical_crossentropy",
                matrics = ["accuracy"])

    # print(model.summary())
    model.fit(x_train, y_train, epochs = 10, batch_size = 100, validation_data = (x_test, y_test))


    pickle.dump(model, open('model.pkl', 'wb'))

if(__name__=='__main__'):
    createModel()