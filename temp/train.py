from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import random as rnd


batch_size = 128
# num_classes - количество классов
num_classes = 10
# epoches - количество эпох обучения
epochs = 5
# процентное соотношение тестировочной выборки ко всей выборки
part_val = 0.27

# размер входных изображений 28*28
img_rows, img_cols = 28, 28

# тренировочные и валидационные данные MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# соединяем тренировочные и валидационные данные MNIST
X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)

# размер общей выборки
size = X.shape[0]

# размер валидационный выборки
val_size = int(size * part_val)

# генерация номеров, которые будут использоваться для валидационной выборки
val_args = []

for x in range(0, val_size):
    i = rnd.randrange(0, size - 1)
    while i in val_args:
        i = rnd.randrange(0, size - 1)
    val_args.append(i)


x_test = X[val_args]
y_test = Y[val_args]

# удаляем валидационные данные из общей выборки
X = np.delete(X, val_args, axis=0)
Y = np.delete(Y, val_args, axis=0)

# reshape
if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# нормализация
X = X.astype('float32')
X /= 255
print('Матрица X обучающих данных имеет размеры: ', X.shape)
print(X.shape[0], 'фрагментов данных для обучения')
print(x_test.shape[0], ' 17% от данных')

# создаем двоичные матрицы классификации
Y = keras.utils.to_categorical(Y, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# определяем модель и слои нейронной сети
model = Sequential()

# добавляем 2-мерный конволюционный слой с 32 нейронами, cетка 3*3 собирают данные с пикселей, стоящих только рядом
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# ещё один конволюционный слой с сеткой 3*3 и 64 нейронами
model.add(Conv2D(64, (3, 3), activation='relu'))

# Pooling слой, обычно ставится после конволюционных, размер сетки 2*2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Если вероятность ниже 0.25, значение этой вероятности становится 0
model.add(Dropout(0.25))

# Flatten-слой позволяет превратить матрицы в строки
model.add(Flatten())

# dense-слой, в котором каждый нейрон связан со всеми входами
model.add(Dense(128, activation='relu'))

# ещё один dropout слой
model.add(Dropout(0.5))

# ещё один dense слой
model.add(Dense(num_classes, activation='softmax'))

# выбираем функцию потери, оптимизатор, а также выводимые метрики
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# X, Y - данные для обучения (X - данные изображений, Y - классы)
# verbose - режим многословия, какие данные будут выводиться при обучении
# validation data - валидационные данные
model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)
          )
score = model.evaluate(x_test, y_test, verbose=0)

print('Точность:', score[1])
model.save("model/model.h5")

