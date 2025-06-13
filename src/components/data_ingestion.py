import pandas as pd
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any
import os
import re

class DataIngestion:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def _convert_duration(self, duration_str):
        """Convert HH:MM:SS or MM:SS to seconds"""
        try:
            if pd.isna(duration_str):
                return 0
            if isinstance(duration_str, (int, float)):
                return float(duration_str)
                
            parts = list(map(float, re.split('[:.]', duration_str)))
            if len(parts) == 3:  # HH:MM:SS
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:  # MM:SS
                return parts[0] * 60 + parts[1]
            return float(duration_str)
        except Exception as e:
            self.logger.warning(f"Could not convert duration {duration_str}: {str(e)}")
            return 0

    def _extract_video_id(self, thumbnail_url):
        """Extract video ID from thumbnail URL or image name"""
        try:
            if pd.isna(thumbnail_url):
                return None
            # Handle both URL and image name formats
            if 'ytimg.com' in thumbnail_url:
                return thumbnail_url.split('/')[-2]  # For URL format
            else:
                return thumbnail_url.split('.')[0]  # For image name format
        except Exception as e:
            self.logger.warning(f"Could not extract video ID: {str(e)}")
            return None

    def merge_data(self):
        try:
            # Load main dataset
            main_path = Path(self.config["raw_data_path"]) / self.config["train_data_file"]
            if not main_path.exists():
                raise FileNotFoundError(f"Training data file not found at {main_path}")
            
            main_df = pd.read_csv(main_path)
            self.logger.info(f"Loaded main dataset with {len(main_df)} records")
            
            # Convert duration to seconds
            main_df['duration_seconds'] = main_df['duration'].apply(self._convert_duration)
            
            # Create time features
            main_df['upload_time'] = pd.to_datetime(main_df['upload_time'])
            main_df['upload_hour'] = main_df['upload_time'].dt.hour
            main_df['upload_dayofweek'] = main_df['upload_time'].dt.dayofweek
            
            # Extract video ID from thumbnail URL
            main_df['video_id_from_thumb'] = main_df['thumbnail'].apply(self._extract_video_id)
            
            # Load thumbnail features
            thumb_path = Path(self.config["raw_data_path"]) / self.config["thumbnail_features_file"]
            if not thumb_path.exists():
                raise FileNotFoundError(f"Thumbnail features file not found at {thumb_path}")
            
            thumb_df = pd.read_csv(thumb_path)
            self.logger.info(f"Loaded thumbnail features with {len(thumb_df)} records")
            
            # Extract video ID from image name
            thumb_df['video_id'] = thumb_df['image_name'].apply(lambda x: x.split('.')[0])
            
            # Merge datasets
            merged_df = pd.merge(
                main_df,
                thumb_df.drop('image_name', axis=1),
                left_on='video_id_from_thumb',
                right_on='video_id',
                how='left'
            )
            
            # Fill missing thumbnail features with defaults
            thumb_cols = ['faces', 'has_text', 'brightness', 'vibrancy', 
                         'contrast', 'color_variety', 'red_dominance',
                         'sharpness', 'aspect_ratio']
            
            for col in thumb_cols:
                if col in merged_df.columns:
                    if col in ['faces', 'has_text']:
                        merged_df[col] = merged_df[col].fillna(0).astype(int)
                    else:
                        merged_df[col] = merged_df[col].fillna(0)
                else:
                    merged_df[col] = 0
                    self.logger.warning(f"Added missing thumbnail feature {col} with 0 values")
            
            # Clean up
            merged_df = merged_df.drop(columns=['video_id_from_thumb', 'video_id_y'], errors='ignore')
            merged_df = merged_df.rename(columns={'video_id_x': 'video_id'})
            
            # Save processed data
            processed_path = Path(self.config["processed_data_path"]) / "processed_data.csv"
            os.makedirs(processed_path.parent, exist_ok=True)
            merged_df.to_csv(processed_path, index=False)
            
            self.logger.info(f"Merged dataset shape: {merged_df.shape}")
            return merged_df
            
        except Exception as e:
            self.logger.error("Error during data ingestion")
            raise