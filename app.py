from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()
predictor = PredictionPipeline()

class PredictionRequest(BaseModel):
    duration: float
    upload_hour: int
    category: str  # Will be encoded in production
    country: str   # Will be encoded in production

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # For testing with minimal features
        input_data = {
            "duration": request.duration,
            "upload_hour": request.upload_hour,
            "category": request.category,
            "country": request.country
        }
        return predictor.predict(input_data)
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "OK"}