from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

class DataTransformation:
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.logger = logging.getLogger(__name__)
        
    def get_preprocessor(self, df: pd.DataFrame):
        """Create preprocessing pipeline with more robust text handling"""
        try:
            # Numeric features
            numeric_features = [
                'duration_seconds', 'upload_hour', 'upload_dayofweek',
                'subscribers', 'faces', 'has_text', 'brightness',
                'vibrancy', 'contrast', 'color_variety',
                'red_dominance', 'sharpness', 'aspect_ratio'
            ]
            
            # Categorical features
            categorical_features = ['category', 'country', 'genre']
            
            # Text features
            text_features = ['title']
            
            # Full preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # More robust text feature handling
            n_samples = len(df)
            max_text_features = min(500, max(10, n_samples // 10))
            
            # More conservative text processing parameters
            text_transformer = Pipeline(steps=[
                ('tfidf', TfidfVectorizer(
                    max_features=max_text_features,
                    stop_words='english',
                    min_df=2,  # Reduced from 5 to 2
                    max_df=0.95,  # Increased from 0.9 to 0.95
                    analyzer='word'
                ))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features),
                    ('text', text_transformer, text_features)
                ],
                remainder='drop'
            )
            
            return preprocessor
            
        except Exception as e:
            self.logger.error(f"Error creating preprocessor: {str(e)}")
            raise
    
    def get_simple_preprocessor(self, df: pd.DataFrame):
        """Create simplified preprocessing pipeline that still supports quantiles"""
        try:
            # Numeric features only but keep it compatible with quantile training
            numeric_features = [
                'duration_seconds', 'upload_hour', 'upload_dayofweek',
                'subscribers', 'faces', 'has_text', 'brightness',
                'vibrancy', 'contrast', 'color_variety',
                'red_dominance', 'sharpness', 'aspect_ratio'
            ]
            numeric_features = [col for col in numeric_features if col in df.columns]
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ],
                remainder='drop'
            )
            
            return preprocessor
            
        except Exception as e:
            self.logger.error(f"Error creating simple preprocessor: {str(e)}")
            raise
    
    def transform_target(self, y: pd.Series):
        """Log transform view_count to handle skew"""
        try:
            return np.log10(y + 1)
        except Exception as e:
            self.logger.error(f"Error transforming target: {str(e)}")
            raise
    
    def inverse_transform(self, y):
        """Convert back from log scale"""
        try:
            return np.power(10, y) - 1
        except Exception as e:
            self.logger.error(f"Error in inverse transform: {str(e)}")
            raise