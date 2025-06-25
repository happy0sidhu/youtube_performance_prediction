from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
import joblib
import numpy as np
from pathlib import Path
import logging

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def train(self, X_train, y_train):
        """Train models with optimized quantile ranges for 30k-50k views"""
        try:
            self.logger.info("Training models for narrow view range (30k-50k)")
            
            # Optimized quantiles for narrow range predictions
            quantiles = [0.4, 0.45, 0.5, 0.55, 0.6]  # Tight range around median
            
            # Optimized LGBM parameters for view count prediction
            lgbm_params = {
                'objective': 'quantile',
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.03,
                'min_child_samples': 30,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1
            }
            
            models = {}
            for q in quantiles:
                self.logger.info(f"Training quantile model (q={q})")
                model = LGBMRegressor(alpha=q, **lgbm_params)
                model.fit(X_train, y_train)
                models[q] = model
            
            self.logger.info(f"Successfully trained {len(models)} quantile models")
            return models
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise

    def evaluate_models(self, models, X_test, y_test):
        """Evaluate models with focus on 30k-50k range accuracy"""
        try:
            metrics = {}
            quantiles = sorted(models.keys())
            
            # Calculate quantile-specific metrics
            for q in quantiles:
                y_pred = models[q].predict(X_test)
                metrics[f"q{int(q*100)}"] = {
                    "pinball": mean_pinball_loss(y_test, y_pred, alpha=q),
                    "mae": mean_absolute_error(y_test, y_pred)
                }
            
            # Calculate range coverage
            lower = models[min(quantiles)].predict(X_test)
            upper = models[max(quantiles)].predict(X_test)
            coverage = np.mean((y_test >= lower) & (y_test <= upper))
            
            metrics["coverage"] = {
                "range": [min(quantiles), max(quantiles)],
                "actual": coverage,
                "expected": max(quantiles) - min(quantiles)
            }
            
            # Median prediction metrics
            if 0.5 in models:
                y_pred = models[0.5].predict(X_test)
                metrics["median"] = {
                    "mae": mean_absolute_error(y_test, y_pred),
                    "in_range": np.mean((y_pred >= 30000) & (y_pred <= 50000))
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

    def save_models(self, models, path: Path):
        """Save trained models with narrow range optimization"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            for q, model in models.items():
                model_path = path / f"model_q{int(q*100)}.pkl"
                joblib.dump(model, model_path)
            self.logger.info(f"Saved {len(models)} models to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")
            raise