from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils.common import read_yaml
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import logging
import sys
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
        
    def _adjust_target_range(self, y):
        """Adjust target values to emphasize our desired range"""
        target_min, target_max = self.config["model_config"]["target_range"]
        center = (target_min + target_max) / 2
        scale = (target_max - target_min) / 2
        
        # Apply tanh scaling to emphasize values near center
        return center + scale * np.tanh((y - center) / scale)
    
    def run(self):
        try:
            # Load config
            config = read_yaml(self.config_path)
            self.config = config
            
            # Data Ingestion
            self.logger.info("=== Starting Data Ingestion ===")
            data_ingestion = DataIngestion(config["data_config"])
            df = data_ingestion.merge_data()
            
            # Data Transformation
            self.logger.info("=== Starting Data Transformation ===")
            X = df.drop(columns=[config["model_config"]["target_column"]])
            y = df[config["model_config"]["target_column"]]
            
            transformer = DataTransformation(config["model_config"]["target_column"])
            y_transformed = transformer.transform_target(y)
            
            # Adjust target range if specified
            if "target_range" in config["model_config"]:
                y_transformed = self._adjust_target_range(y_transformed)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_transformed,
                test_size=config["model_config"]["test_size"],
                random_state=config["model_config"]["random_state"]
            )
            
            # Preprocessing
            preprocessor = transformer.get_preprocessor(X_train)
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Save preprocessor
            joblib.dump(preprocessor, Path("artifacts/preprocessor.pkl"))
            
            # Model Training
            self.logger.info("=== Starting Model Training ===")
            model_trainer = ModelTrainer(config)
            models = model_trainer.train(X_train_processed, y_train)
            
            # Save models
            model_trainer.save_models(models, Path("artifacts/models"))
            
            # Evaluation
            metrics = model_trainer.evaluate_models(models, X_test_processed, y_test)
            self.logger.info(f"Final metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise