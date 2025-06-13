from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig, 
                                     ModelTrainerConfig,
                                     PredictionConfig)
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath="configs/config.yaml"):
        self.config = read_yaml(config_filepath)
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_config
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path("artifacts/data_ingestion"),
            raw_data_path=Path(config.raw_data_path),
            processed_data_path=Path(config.processed_data_path),
            train_data_file=config.train_data_file,
            thumbnail_features_file=config.thumbnail_features_file
        )
        
        return data_ingestion_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_config
        train_config = self.config.train_config
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path("artifacts/model_trainer"),
            model_dir=Path("models"),
            target_column=config.target_column,
            test_size=config.test_size,
            random_state=config.random_state,
            quantiles=config.quantiles,
            rf_params=train_config.rf_params,
            lgbm_params=train_config.lgbm_params
        )
        
        return model_trainer_config
    
    def get_prediction_config(self) -> PredictionConfig:
        return PredictionConfig(
            model_dir=Path("models"),
            preprocessor_path=Path("artifacts/preprocessor.pkl")
        )