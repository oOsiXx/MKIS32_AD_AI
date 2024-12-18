Прогнозирование оттока клиентов телеком компании
Этот проект представляет собой веб-приложение, которое использует модель машинного обучения для прогнозирования вероятности оттока клиентов телеком компании. Приложение позволяет загружать .csv файл с данными клиентов и получать прогнозы для каждого клиента.

Как запустить:
 1. Установите зависимости:
"npm install @tensorflow/tfjs-node express body-parser multer csv-parser"
 2. Запустите сервер:
"node server.js"
 3. Откройте браузер и перейдите по адресу http://localhost:3000.

 Как использовать
  1. Загрузите .csv файл:
    - Откройте браузер и перейдите по адресу http://localhost:3000.
    - На странице будет форма для загрузки .csv файла. Файл должен содержать данные клиентов в формате, где каждая строка представляет клиента, а каждый столбец — характеристику клиента (всего 19 чисел в каждой строке).
  2. Получите прогноз:
    - После загрузки файла нажмите кнопку "Получить прогноз".
    - Сервер обработает файл, сделает прогноз и вернет вероятности для каждого клиента.
Результаты будут отображены на странице.

Технологии
  - TensorFlow.js: Используется для загрузки и работы с моделью машинного обучения.
  - Express.js: Веб-сервер для обработки запросов.
  - Multer: Для обработки загрузки файлов.
  - CSV-Parser: Для чтения данных из .csv файлов.
