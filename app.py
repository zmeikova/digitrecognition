"""
app.py — FastAPI REST-сервис для распознавания рукописных цифр.

Запуск:
    uvicorn app:app --reload

Swagger UI: http://127.0.0.1:8000/docs
"""

import io
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, field_validator
from PIL import Image

app = FastAPI(
    title="Digit Recognition Service",
    description="REST-API для распознавания рукописных цифр (sklearn + SVM).",
    version="1.0.0",
)

MODEL_PATH = "models/model_pipeline.pkl"

try:
    _artifacts = joblib.load(MODEL_PATH)
    pipeline   = _artifacts["model"]
    metrics    = _artifacts.get("metrics", {})
except FileNotFoundError:
    raise RuntimeError(
        f"Модель не найдена: '{MODEL_PATH}'. "
        "Сначала запустите: python train.py"
    )

class DigitRequest(BaseModel):
    pixels: List[float]

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, v: List[float]) -> List[float]:
        if len(v) != 64:
            raise ValueError(f"Ожидается 64 значения (8×8), получено {len(v)}.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {"pixels": [0.0] * 64}
        }
    }


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    needs_manual_review: bool

def image_to_pixels(data: bytes) -> np.ndarray:
    """Конвертирует загруженное изображение в вектор 64 значений (0–16)."""
    img = Image.open(io.BytesIO(data)).convert("L")
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float64)
    # Нормируем в диапазон [0, 16], как в sklearn digits
    arr = arr / 255.0 * 16.0
    return arr.flatten()

@app.get("/health", summary="Проверка работоспособности")
def health():
    """Возвращает статус сервиса и метрики обученной модели."""
    return {"status": "ok", "model_loaded": True, "metrics": metrics}


@app.get("/info", summary="Информация о сервисе")
def info():
    return {
        "service": "Digit Recognition Service",
        "version": "1.0.0",
        "description": "Распознавание рукописных цифр (0–9) на изображениях 8×8 пикселей.",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Предсказание по вектору пикселей",
)
def predict_json(request: DigitRequest):
    """
    Принимает JSON с массивом из 64 значений интенсивности пикселей (0–16).
    Возвращает предсказанную цифру и уверенность модели.
    """
    x = np.array(request.pixels, dtype=np.float64).reshape(1, -1)
    digit      = int(pipeline.predict(x)[0])
    confidence = float(pipeline.predict_proba(x)[0].max())

    return PredictionResponse(
        digit=digit,
        confidence=round(confidence, 4),
        needs_manual_review=confidence < 0.85,
    )


@app.post(
    "/predict/upload",
    response_model=PredictionResponse,
    summary="Предсказание по загруженному изображению",
)
async def predict_upload(file: UploadFile = File(...)):
    """
    Принимает изображение (PNG/JPEG) произвольного размера.
    Автоматически масштабирует до 8×8 и возвращает предсказание.
    """
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый тип файла: {file.content_type}. "
                   f"Поддерживаются: PNG, JPEG, BMP.",
        )

    contents = await file.read()
    try:
        pixels = image_to_pixels(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка обработки изображения: {e}")

    x = pixels.reshape(1, -1)
    digit      = int(pipeline.predict(x)[0])
    confidence = float(pipeline.predict_proba(x)[0].max())

    return PredictionResponse(
        digit=digit,
        confidence=round(confidence, 4),
        needs_manual_review=confidence < 0.85,
    )
