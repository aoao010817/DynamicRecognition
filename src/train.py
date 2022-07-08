import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Reshape, MaxPooling2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
import numpy as np

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

hist = model.fit(X_train, y_train, batch_size=200, verbose=1, 
                 epochs=10, validation_split=0.1)

#評価
score = model.evaluate(X_test, y_test, verbose=1)
print("正解率(acc)：", score[1])

model.save("MNIST.h5")