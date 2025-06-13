import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

class PredictionPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = self._load_models()
        self.preprocessor = self._load_preprocessor()
        self.quantiles = self._get_quantiles()
        
    def _load_models(self):
        """Load all available quantile models"""
        models = {}
        model_dir = Path("artifacts/models")
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {model_dir}")
        
        for model_file in model_dir.glob("model_q*.pkl"):
            try:
                quantile = int(model_file.stem.split('q')[1]) / 100
                models[quantile] = joblib.load(model_file)
                self.logger.info(f"Loaded model for quantile {quantile}")
            except Exception as e:
                self.logger.warning(f"Could not load model {model_file}: {str(e)}")
        
        if not models:
            raise ValueError("No valid models found in artifacts/models directory")
        
        return models
    
    def _load_preprocessor(self):
        """Load the preprocessor"""
        preprocessor_path = Path("artifacts/preprocessor.pkl")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        return joblib.load(preprocessor_path)
    
    def _get_quantiles(self):
        """Get sorted list of available quantiles"""
        return sorted(self.models.keys())
    
    def predict(self, data: dict):
        """Make quantile predictions for input data"""
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([data])
            
            # Preprocess
            processed_input = self.preprocessor.transform(input_df)
            
            # Predict for all quantiles
            predictions = {}
            for q in self.quantiles:
                pred = self.models[q].predict(processed_input)[0]
                predictions[q] = np.power(10, pred) - 1  # Convert from log scale
            
            # Prepare response
            response = {
                "predictions": {f"q{int(q*100)}": int(predictions[q]) for q in self.quantiles},
                "median_views": int(predictions.get(0.5, 0)),
                "uncertainty": self._calculate_uncertainty(predictions)
            }
            
            # Add prediction interval if we have multiple quantiles
            if len(self.quantiles) >= 2:
                lower_q = min(self.quantiles)
                upper_q = max(self.quantiles)
                response["prediction_range"] = {
                    "lower": int(predictions[lower_q]),
                    "upper": int(predictions[upper_q]),
                    "confidence": f"{int((upper_q - lower_q)*100)}%"
                }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _calculate_uncertainty(self, predictions):
        """Calculate uncertainty metrics based on quantile predictions"""
        if 0.05 in predictions and 0.95 in predictions:
            return {
                "range_90": int(predictions[0.95] - predictions[0.05]),
                "relative_uncertainty": (predictions[0.95] - predictions[0.05]) / max(1, predictions.get(0.5, 1))
            }
        elif 0.1 in predictions and 0.9 in predictions:
            return {
                "range_80": int(predictions[0.9] - predictions[0.1]),
                "relative_uncertainty": (predictions[0.9] - predictions[0.1]) / max(1, predictions.get(0.5, 1))
            }
        elif 0.25 in predictions and 0.75 in predictions:
            return {
                "range_50": int(predictions[0.75] - predictions[0.25]),
                "relative_uncertainty": (predictions[0.75] - predictions[0.25]) / max(1, predictions.get(0.5, 1))
            }
        return {"warning": "Limited uncertainty estimation available"}