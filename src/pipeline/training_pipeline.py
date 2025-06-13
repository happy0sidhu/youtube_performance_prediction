from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_pinball_loss
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.utils.common import read_yaml
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import logging
import sys
import os
import joblib

class TrainingPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        try:
            # Load config
            config = read_yaml(self.config_path)
            
            # Data Ingestion
            self.logger.info("=== Starting Data Ingestion ===")
            data_ingestion = DataIngestion(config["data_config"])
            df = data_ingestion.merge_data()
            
            # Check if we have enough samples
            min_samples = config["data_config"].get("min_samples_required", 10)
            if len(df) < min_samples:
                self.logger.warning(f"Warning: Only {len(df)} samples available (minimum {min_samples} recommended)")
                if len(df) < 2:
                    raise ValueError("Insufficient samples for training (need at least 2)")
            
            # Data Transformation
            self.logger.info("=== Starting Data Transformation ===")
            X = df.drop(columns=[config["model_config"]["target_column"]])
            y = df[config["model_config"]["target_column"]]
            
            transformer = DataTransformation(config["model_config"]["target_column"])
            y_transformed = transformer.transform_target(y)
            
            # Handle small datasets differently
            if len(df) < 10 and config["model_config"].get("skip_train_test_split_if_small", False):
                self.logger.warning("Small dataset - skipping train-test split")
                X_train, X_test, y_train, y_test = X, X, y_transformed, y_transformed
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_transformed,
                    test_size=config["model_config"]["test_size"],
                    random_state=config["model_config"]["random_state"]
                )
            
            # Attempt full preprocessing first
            self.logger.info("=== Attempting Full Preprocessing ===")
            try:
                preprocessor = transformer.get_preprocessor(X_train)
                X_train_preprocessed = preprocessor.fit_transform(X_train)
                X_test_preprocessed = preprocessor.transform(X_test)
                
                # Save preprocessor
                preprocessor_path = Path("artifacts/preprocessor.pkl")
                joblib.dump(preprocessor, preprocessor_path)
                self.logger.info(f"Saved preprocessor to {preprocessor_path}")
                
                self.logger.info(f"Full preprocessing successful. Training shape: {X_train_preprocessed.shape}")
                
                # Model Training
                model_trainer = ModelTrainer(config)
                models = model_trainer.train(X_train_preprocessed, y_train)
                
                # Save models
                model_trainer.save_models(models, Path("artifacts/models"))
                
                # Evaluation
                metrics = model_trainer.evaluate_models(models, X_test_preprocessed, y_test)
                
                self.logger.info("=== Training Completed Successfully with Full Features ===")
                self.logger.info(f"Metrics: {metrics}")
                return metrics
                
            except Exception as e:
                self.logger.error(f"Full preprocessing failed: {str(e)}")
                self.logger.warning("Falling back to simplified preprocessing...")
                
                # Simplified preprocessing with just numeric features
                simple_preprocessor = transformer.get_simple_preprocessor(X_train)
                X_train_simple = simple_preprocessor.fit_transform(X_train)
                X_test_simple = simple_preprocessor.transform(X_test)
                
                # Save simple preprocessor
                preprocessor_path = Path("artifacts/preprocessor.pkl")
                joblib.dump(simple_preprocessor, preprocessor_path)
                self.logger.info(f"Saved simple preprocessor to {preprocessor_path}")
                
                self.logger.info(f"Using simplified preprocessing with {X_train_simple.shape[1]} numeric features")
                
                # Train with simplified features (only median)
                model_trainer = ModelTrainer(config)
                models = {0.5: model_trainer.train_rf_model(X_train_simple, y_train)}
                
                # Save models
                model_trainer.save_models(models, Path("artifacts/models"))
                
                # Evaluation
                metrics = model_trainer.evaluate_models(models, X_test_simple, y_test)
                metrics["mode"] = "simplified"
                metrics["warning"] = "Only median prediction available in simplified mode"
                
                self.logger.info("=== Training Completed with Simplified Features ===")
                self.logger.info(f"Metrics: {metrics}")
                return metrics
                
        except Exception as e:
            self.logger.error("!!! Pipeline Failed !!!")
            self.logger.error(f"Error: {str(e)}")
            raise