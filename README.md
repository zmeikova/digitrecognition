# Digit Recognition Service

REST-API для автоматического распознавания рукописных цифр (0–9).

**Стек:** Python · scikit-learn (SVM) · FastAPI · Pillow · joblib

---

## Структура проекта

```
digit-recognition/
├── app.py              # FastAPI приложение
├── requirements.txt    # Зависимости
├── README.md
├── .gitignore
├── models/
│   └── model_pipeline.pkl   # Веса модели (обучены в Google Colab)
└── plots/
    └── confusion_matrix.png # График качества модели
```

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/swirthuk/digit-recognition.git
cd digit-recognition
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Положить модель в папку `models/`

Модель обучается в Google Colab. После обучения скачайте файл весов:

```python
# В конце Colab-ноутбука:
from google.colab import files
files.download("models/model_pipeline.pkl")
```

Скачанный файл `model_pipeline.pkl` положите в папку `models/`.

### 4. Запустить сервер локально

```bash
uvicorn app:app --reload
```

Сервер доступен по адресу: **http://127.0.0.1:8000**  
Swagger UI: **http://127.0.0.1:8000/docs**

---

## API

### `GET /health`
Проверка работоспособности сервиса.

```json
{
  "status": "ok",
  "model_loaded": true,
  "metrics": {"accuracy": 0.9778, "f1_macro": 0.9776}
}
```

### `POST /predict`
Предсказание по вектору из 64 пикселей (JSON).

**Запрос:**
```json
{ "pixels": [0, 0, 5, 13, 9, 1, 0, 0, ...] }
```

**Ответ:**
```json
{
  "digit": 3,
  "confidence": 0.9821,
  "needs_manual_review": false
}
```

### `POST /predict/upload`
Предсказание по изображению (PNG / JPEG / BMP). Автоматически масштабируется до 8×8.

```bash
curl -X POST http://127.0.0.1:8000/predict/upload \
     -F "file=@digit.png"
```

---

## Метрики модели (SVM, тестовая выборка)

| Метрика  | Значение |
|----------|----------|
| Accuracy | 0.9778   |
| F1-macro | 0.9776   |

Модель обучена на `sklearn.datasets.load_digits` (1 797 изображений, 8×8 пикселей, 10 классов).
