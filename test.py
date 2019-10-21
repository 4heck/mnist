from __future__ import print_function
import keras
import json
import numpy as np
from keras import backend as K
from PIL import Image
from sklearn import metrics

# загружаем натренированную сеть
model = keras.models.load_model("model/model.h5")

# картинки 28*28
img_rows, img_cols = 28, 28

# инициализируем лист с картинками
data = []
for i in range(10):
    img = Image.open("data/" + str(i) + ".png")
    temp = list(img.getdata())
    data.append(temp)
data = np.array(data)

# reshape
if K.image_data_format() == 'channels_first':
    data = data.reshape(data.shape[0], 1, img_rows, img_cols)
else:
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)

# нормализируем
data = data.astype('float32')
data /= 255

# выполняем классификацию
pred = model.predict(data)
values = pred.argmax(axis=0).tolist()
pred = pred.tolist()

json_list = []

for i in range(10):
    x = {"Ожидание": i, "Предположение": values[i], "Вероятностный вектор результата": pred[i]}
    json_list.append(x)

with open("result.json", 'w', encoding='utf-8') as f:
    json.dump(json_list, f, ensure_ascii=False, indent=4)

y_test = []
for i in range(10):
    y_test.append(i)

matrix = metrics.confusion_matrix(y_test, values)
print(matrix)
