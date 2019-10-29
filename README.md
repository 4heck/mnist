**Описание скриптов:**
 - _train.py_ - загрузка данных из базы данных keras.mnist, создание нейросети и обучение на этих данных.
 - _test.py_ - загрузка обученной модели и проверка на картинках из data (0 - 9.png), выдача результата (result.json) загрузка данных из базы данных и создание confusion-матрицы.
 - _confusion.py_ - загрузка обученной модели из model/model.h5 и создание confusion-матрицы.

**Создание и активация виртуальной среды:**
 - `python3 -m venv venv`
 - `source venv/bin/activate`
 
**Установка зависимостей:**
 - `pip install -r requirements.txt`
 
**Запуск скриптов:**
 - `python3 train.py`
 - `python3 test.py`
 - `python3 confusion.py`
 
 
 
