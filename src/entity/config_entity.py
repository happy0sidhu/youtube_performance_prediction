from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    raw_data_path: Path
    processed_data_path: Path
    train_data_file: str
    thumbnail_features_file: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_dir: Path
    target_column: str
    test_size: float
    random_state: int
    quantiles: list
    rf_params: dict
    lgbm_params: dict

@dataclass(frozen=True)
class PredictionConfig:
    model_dir: Path
    preprocessor_path: Path