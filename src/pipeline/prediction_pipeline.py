import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

class PredictionPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = self._load_models()
        self.preprocessor = self._load_preprocessor()
    
    def _load_models(self):
        models = {}
        model_dir = Path("artifacts/models")
        
        for model_file in model_dir.glob("model_q*.pkl"):
            quantile = int(model_file.stem.split('q')[1]) / 100
            models[quantile] = joblib.load(model_file)
            self.logger.info(f"Loaded model for quantile {quantile}")
        
        if not models:
            raise ValueError("No models found in artifacts/models")
        
        return models
    
    def _load_preprocessor(self):
        preprocessor_path = Path("artifacts/preprocessor.pkl")
        if not preprocessor_path.exists():
            raise FileNotFoundError("Preprocessor not found")
        return joblib.load(preprocessor_path)
    
    def predict(self, input_data: dict) -> dict:
        input_df = pd.DataFrame([input_data])
        processed_input = self.preprocessor.transform(input_df)
        
        raw_preds = {
            q: self._inverse_transform(model.predict(processed_input)[0])
            for q, model in self.models.items()
        }
        
        median_pred = raw_preds.get(0.5, 0)
        
        return {
            "median_views": int(median_pred),
            "prediction_range": {
                "lower": 0,
                "upper": 0,
                "confidence": "N/A"
            },
            "is_in_target_range": False
        }
        
    def _inverse_transform(self, y):
        return np.power(10, y) - 1
    
    def _constrain_to_range(self, value):
        return value