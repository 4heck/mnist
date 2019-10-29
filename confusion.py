from __future__ import print_function
import keras
from keras import backend as K
from keras.datasets import mnist
from sklearn import metrics

num_classes = 10
img_rows, img_cols = 28, 28

model = keras.models.load_model("model/model.h5")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

y_test = keras.utils.to_categorical(y_test, num_classes)

score = model.evaluate(x_test, y_test, verbose=0)
print('Потеря:', score[0])
print('Точность:', score[1])

# с помощью обученной сетки получаем выборку в виде набора классов
y_pred = model.predict(x_test)
# создаём непосредственно confusion матрицу
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)