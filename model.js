const tf = require('@tensorflow/tfjs-node');
const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer'); // Для обработки файлов
const csv = require('csv-parser'); // Для чтения CSV
const fs = require('fs');
const path = require('path');

const app = express();
app.use(bodyParser.json());

// Настройка multer для загрузки файлов
const upload = multer({ dest: 'uploads/' });

// Загрузка модели
let model;
async function loadModel() {
    model = await tf.loadLayersModel('file://model.json');
    console.log('Модель загружена');
}
loadModel();

// Маршрут для загрузки CSV и прогнозирования
app.post('/predict', upload.single('file'), async (req, res) => {
    const filePath = req.file.path;

    const data = [];
    fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (row) => {
            // Преобразуем строку в массив чисел
            const features = Object.values(row).map(Number);
            if (features.length === 19) {
                data.push(features);
            }
        })
        .on('end', async () => {
            if (data.length === 0) {
                return res.status(400).json({ error: 'CSV файл должен содержать 19 чисел в каждой строке.' });
            }

            // Преобразуем данные в тензор
            const inputTensor = tf.tensor2d(data);

            // Делаем прогноз
            const predictions = model.predict(inputTensor);
            const probabilities = await predictions.array();

            // Удаляем временный файл
            fs.unlinkSync(filePath);

            // Возвращаем результат
            res.json({ probabilities });
        });
});

// Маршрут для статического HTML
app.use(express.static(path.join(__dirname, 'public')));

// Запуск сервера
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Сервер запущен на http://localhost:${PORT}`);
});