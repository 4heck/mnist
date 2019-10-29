from __future__ import print_function
import keras
from keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# загружаем натренированную сеть
model = keras.models.load_model("model/model.h5")

# картинки 28*28
img_rows, img_cols = 28, 28

# инициализируем лист с картинками
data = []

# i - цифра, которую нужно проверить
i = 0
img = Image.open("data/" + str(i) + ".png")
temp = list(img.getdata())
temp = np.array(temp)

print(temp.shape)
print(np.max(temp))
data.append(temp)
data = np.array(data)


# reshape
if K.image_data_format() == 'channels_first':
    data = data.reshape(data.shape[0], 1, img_rows, img_cols)
else:
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)

# нормализируем
data1 = data.astype('float32')
data = data1/255.0
x_train1 = data.reshape(28, 28)
plt.imshow(x_train1)
plt.show()

# выполняем классификацию
pred = model.predict(data)
print("Type pred", pred)
values = pred.argmax(axis=0).tolist()
pred = pred.tolist()
print("Type pred", pred)
json_list = []


arr = [0,1,2,3,4,5,6,7,8,9]
print(arr)
print(pred[0])
plt.xticks(arr, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
plt.bar(arr, pred[0])
plt.show()
