from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_pinball_loss
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List
from src.utils.common import save_json
import logging

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def train(self, X_train, y_train):
        """Main training method that trains quantile regression models"""
        try:
            self.logger.info("Starting quantile model training")
            
            quantiles = self.config["model_config"].get("quantiles", [0.5])
            models = {}
            
            # Adjust parameters based on dataset size
            n_samples = X_train.shape[0]
            lgbm_config = self._adjust_lgbm_params(n_samples)
            
            # Train LightGBM models for all quantiles
            for q in quantiles:
                self.logger.info(f"Training model for quantile: {q}")
                
                lgbm_model = LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    **lgbm_config,
                    verbose=-1
                )
                lgbm_model.fit(X_train, y_train)
                models[q] = lgbm_model
                self.logger.info(f"Trained LightGBM for quantile {q}")
            
            self.logger.info(f"Trained {len(models)} quantile models")
            return models
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
    
    def train_simplified(self, X_train, y_train):
        """Train simplified models (still with quantiles)"""
        try:
            self.logger.info("Starting simplified quantile model training")
            
            quantiles = self.config["model_config"].get("quantiles", [0.5])
            models = {}
            
            # Use only LightGBM for simplified training
            lgbm_config = {
                "n_estimators": 50,
                "max_depth": 5,
                "learning_rate": 0.1
            }
            
            for q in quantiles:
                self.logger.info(f"Training simplified model for quantile: {q}")
                
                lgbm_model = LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    **lgbm_config,
                    verbose=-1
                )
                lgbm_model.fit(X_train, y_train)
                models[q] = lgbm_model
                self.logger.info(f"Trained simplified LightGBM for quantile {q}")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error in simplified model training: {str(e)}")
            raise
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate model performance with quantile-specific metrics"""
        try:
            metrics = {}
            quantiles = sorted(models.keys())
            
            # Calculate metrics for each quantile
            for q in quantiles:
                model = models[q]
                y_pred = model.predict(X_test)
                
                # Store quantile-specific metrics
                metrics[f"quantile_{q}"] = {
                    "pinball_loss": mean_pinball_loss(y_test, y_pred, alpha=q),
                    "mae": mean_absolute_error(y_test, y_pred)
                }
            
            # Calculate coverage metrics if we have multiple quantiles
            if len(quantiles) >= 2:
                lower_q = min(quantiles)
                upper_q = max(quantiles)
                
                lower = models[lower_q].predict(X_test)
                upper = models[upper_q].predict(X_test)
                
                coverage = np.mean((y_test >= lower) & (y_test <= upper))
                metrics["coverage"] = {
                    "range": [lower_q, upper_q],
                    "actual_coverage": coverage,
                    "expected_coverage": upper_q - lower_q
                }
            
            # For median (q=0.5), calculate traditional metrics
            if 0.5 in models:
                y_pred_median = models[0.5].predict(X_test)
                metrics["median"] = {
                    "mae": mean_absolute_error(y_test, y_pred_median),
                    "r2": r2_score(y_test, y_pred_median)
                }
            
            save_json(Path("artifacts/metrics.json"), metrics)
            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            raise
    
    def save_models(self, models, path: Path):
        """Save trained models to disk"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            for q, model in models.items():
                model_path = path / f"model_q{int(q*100)}.pkl"
                joblib.dump(model, model_path)
                self.logger.info(f"Saved model to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
    
    def _adjust_lgbm_params(self, n_samples):
        """Adjust LightGBM parameters based on sample size"""
        config = self.config["train_config"]["lightgbm"].copy()
        if n_samples < 100:
            config["n_estimators"] = min(50, config["n_estimators"])
            config["max_depth"] = min(3, config["max_depth"])
        return config