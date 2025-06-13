# # import os
# # import sys
# # import pandas as pd
# # import numpy as np
# # import re
# # from dataclasses import dataclass
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.preprocessing import OneHotEncoder
# # from src.exception import CustomException
# # from src.logger import logging
# # from pathlib import Path
# # import nltk
# # from nltk.corpus import stopwords
# # from nltk.stem import WordNetLemmatizer
# # from nltk.tokenize import word_tokenize
# # from textblob import TextBlob
# # import emoji
# # from collections import Counter

# # # Download NLTK resources (only needed once)
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('wordnet')

# # @dataclass
# # class TextProcessingConfig:
# #     # Use absolute paths to be sure
# #     preprocessed_data_path: str = os.path.abspath(os.path.join('artifacts', 'text_features.csv'))
# #     raw_data_path: str = os.path.abspath(os.path.join('data', 'youtube_dataset_with_full_countries.csv'))
# # class TextDataPreprocessor:
# #     def __init__(self):
# #         self.config = TextProcessingConfig()
# #         self.lemmatizer = WordNetLemmatizer()
# #         self.stop_words = set(stopwords.words('english'))
# #         # Add YouTube-specific stop words
# #         self.stop_words.update(['youtube', 'video', 'watch', 'channel', 'subscribe', 'like', 'share'])
        
# #     def preprocess_text_data(self):
# #         try:
# #             logging.info("Starting text data preprocessing")
            
# #             # Load raw data
# #             df = pd.read_csv(self.config.raw_data_path)
# #             logging.info(f"Loaded raw data with {len(df)} rows")
            
# #             # Basic cleaning
# #             df = self._clean_data(df)
            
# #             # Extract text features
# #             text_features = self._extract_text_features(df['title'])
            
# #             # Combine with metadata
# #             metadata_features = self._extract_metadata_features(df)
# #             final_df = pd.concat([text_features, metadata_features], axis=1)
            
# #             # Save processed data
# #             self._save_data(final_df)
# #             logging.info("Text preprocessing completed successfully")
            
# #             return final_df
            
# #         except Exception as e:
# #             logging.error(f"Error in text preprocessing: {str(e)}")
# #             raise CustomException(e, sys)
    
# #     def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
# #         """Perform basic data cleaning"""
# #         # Drop duplicates
# #         df = df.drop_duplicates(subset=['title'])
        
# #         # Handle missing values
# #         df['title'] = df['title'].fillna('')
        
# #         # Clean title text
# #         df['title'] = df['title'].apply(self._clean_text)
        
# #         return df
    
# #     def _clean_text(self, text: str) -> str:
# #         """Clean and normalize text"""
# #         if not isinstance(text, str):
# #             return ""
            
# #         # Convert to lowercase
# #         text = text.lower()
        
# #         # Remove special characters but keep basic punctuation
# #         text = re.sub(r'[^\w\s.,!?]', '', text)
        
# #         # Remove extra whitespace
# #         text = ' '.join(text.split())
        
# #         return text
    
# #     def _extract_text_features(self, text_series: pd.Series) -> pd.DataFrame:
# #         """Extract NLP features from titles"""
# #         features = {}
        
# #         # Basic text features
# #         features['title_length'] = text_series.apply(len)
# #         features['word_count'] = text_series.apply(lambda x: len(x.split()))
# #         features['avg_word_length'] = text_series.apply(
# #             lambda x: np.mean([len(w) for w in x.split()]) if x else 0
# #         )
        
# #         # Sentiment analysis
# #         features['sentiment_polarity'] = text_series.apply(
# #             lambda x: TextBlob(x).sentiment.polarity
# #         )
# #         features['sentiment_subjectivity'] = text_series.apply(
# #             lambda x: TextBlob(x).sentiment.subjectivity
# #         )
        
# #         # Structural features
# #         features['has_question'] = text_series.str.contains(r'\?').astype(int)
# #         features['has_exclamation'] = text_series.str.contains(r'\!').astype(int)
# #         features['has_number'] = text_series.str.contains(r'\d').astype(int)
# #         features['has_emoji'] = text_series.apply(lambda x: int(any(char in emoji.UNICODE_EMOJI for char in x)))
        
# #         # Keyword features
# #         features['has_power_word'] = text_series.apply(self._contains_power_word)
# #         features['clickbait_score'] = text_series.apply(self._calculate_clickbait_score)
        
# #         # Topic modeling placeholder (you can expand this)
# #         features['is_movie_related'] = text_series.str.contains(
# #             r'movie|film|cinema|actor|actress|director|trailer', 
# #             case=False
# #         ).astype(int)
        
# #         # Tokenization and lemmatization for later use
# #         processed_text = text_series.apply(self._tokenize_and_lemmatize)
# #         features['processed_text'] = processed_text
        
# #         return pd.DataFrame(features)
    
# #     def _extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
# #         """Extract features from other columns"""
# #         features = {}
        
# #         # Duration features
# #         df['duration'] = pd.to_timedelta(df['duration'])
# #         features['duration_seconds'] = df['duration'].dt.total_seconds()
        
# #         # Category features (one-hot encoded)
# #         encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# #         category_encoded = encoder.fit_transform(df[['category']])
# #         category_df = pd.DataFrame(
# #             category_encoded,
# #             columns=[f"category_{cat}" for cat in encoder.categories_[0]]
# #         )
        
# #         # Country features
# #         features['is_india'] = (df['country'] == 'India').astype(int)
# #         features['is_us'] = (df['country'] == 'United Sta').astype(int)  # Note: Fix data typo
        
# #         # Subscriber count (log transformed)
# #         features['log_subscribers'] = np.log1p(df['subscriber:'])
        
# #         # Combine all metadata features
# #         metadata_df = pd.DataFrame(features)
# #         return pd.concat([metadata_df, category_df], axis=1)
    
# #     def _tokenize_and_lemmatize(self, text: str) -> str:
# #         """Tokenize and lemmatize text for later vectorization"""
# #         tokens = word_tokenize(text)
# #         lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens 
# #                      if token not in self.stop_words and token.isalpha()]
# #         return ' '.join(lemmatized)
    
# #     def _contains_power_word(self, text: str) -> int:
# #         """Check if text contains common power words"""
# #         power_words = [
# #             'secret', 'proven', 'ultimate', 'best', 'top', 'amazing',
# #             'incredible', 'essential', 'perfect', 'complete', 'advanced',
# #             'exclusive', 'instant', 'quick', 'easy', 'guaranteed'
# #         ]
# #         return int(any(word in text for word in power_words))
    
# #     def _calculate_clickbait_score(self, text: str) -> float:
# #         """Calculate a simple clickbait score"""
# #         clickbait_phrases = [
# #             r'you won\'t believe', r'shocked to see', r'what happens next',
# #             r'before you watch', r'the truth about', r'going viral',
# #             r'blow your mind', r'can\'t be unseen'
# #         ]
# #         score = 0
# #         for phrase in clickbait_phrases:
# #             if re.search(phrase, text, re.IGNORECASE):
# #                 score += 1
# #         return score / len(clickbait_phrases)  # Normalized score
    
# #     def _save_data(self, df: pd.DataFrame):
# #         """Save processed data to CSV"""
# #         os.makedirs(os.path.dirname(self.config.preprocessed_data_path), exist_ok=True)
# #         df.to_csv(self.config.preprocessed_data_path, index=False)
# #         logging.info(f"Saved processed text data to {self.config.preprocessed_data_path}")

# # if __name__ == "__main__":
# #     try:
# #         preprocessor = TextDataPreprocessor()
# #         processed_data = preprocessor.preprocess_text_data()
# #         print(processed_data.head())
# #     except Exception as e:
# #         print(f"Error: {str(e)}")




# import os
# import sys
# import pandas as pd
# import numpy as np
# import re
# from dataclasses import dataclass
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from src.exception import CustomException
# from src.logger import logging
# from pathlib import Path
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from textblob import TextBlob
# import emoji
# from collections import Counter

# # Add project root to Python path
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

# # Download NLTK resources
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

# @dataclass
# class TextProcessingConfig:
#     preprocessed_data_path: str = os.path.join('artifacts', 'text_features.csv')
#     raw_data_path: str = os.path.join('data', 'youtube_dataset_with_full_countries.csv')

# class TextDataPreprocessor:
#     def __init__(self):
#         self.config = TextProcessingConfig()
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words = set(stopwords.words('english'))
#         # Add YouTube-specific stop words
#         self.stop_words.update(['youtube', 'video', 'watch', 'channel', 'subscribe', 'like', 'share'])
        
#     def preprocess_text_data(self):
#         try:
#             logging.info("Starting text data preprocessing")
            
#             # Load raw data
#             df = pd.read_csv(self.config.raw_data_path)
#             logging.info(f"Loaded raw data with {len(df)} rows")
            
#             # Basic cleaning
#             df = self._clean_data(df)
            
#             # Extract text features
#             text_features = self._extract_text_features(df['title'])
            
#             # Combine with metadata
#             metadata_features = self._extract_metadata_features(df)
#             final_df = pd.concat([text_features, metadata_features], axis=1)
            
#             # Save processed data
#             self._save_data(final_df)
#             logging.info("Text preprocessing completed successfully")
            
#             return final_df
            
#         except Exception as e:
#             logging.error(f"Error in text preprocessing: {str(e)}")
#             raise CustomException(e, sys)
    
#     def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Perform basic data cleaning"""
#         # Create a copy to avoid SettingWithCopyWarning
#         df = df.copy()
        
#         # Drop duplicates
#         df = df.drop_duplicates(subset=['title']).copy()
        
#         # Handle missing values
#         df.loc[:, 'title'] = df['title'].fillna('')
        
#         # Clean title text
#         df.loc[:, 'title'] = df['title'].apply(self._clean_text)
        
#         return df
    
#     def _clean_text(self, text: str) -> str:
#         """Clean and normalize text"""
#         if not isinstance(text, str):
#             return ""
            
#         # Convert to lowercase
#         text = text.lower()
        
#         # Remove special characters but keep basic punctuation
#         text = re.sub(r'[^\w\s.,!?]', '', text)
        
#         # Remove extra whitespace
#         text = ' '.join(text.split())
        
#         return text
    
#     def _extract_text_features(self, text_series: pd.Series) -> pd.DataFrame:
#         """Extract NLP features from titles"""
#         features = {}
        
#         # Basic text features
#         features['title_length'] = text_series.apply(len)
#         features['word_count'] = text_series.apply(lambda x: len(x.split()))
#         features['avg_word_length'] = text_series.apply(
#             lambda x: np.mean([len(w) for w in x.split()]) if x else 0
#         )
        
#         # Sentiment analysis
#         features['sentiment_polarity'] = text_series.apply(
#             lambda x: TextBlob(x).sentiment.polarity
#         )
#         features['sentiment_subjectivity'] = text_series.apply(
#             lambda x: TextBlob(x).sentiment.subjectivity
#         )
        
#         # Structural features
#         features['has_question'] = text_series.str.contains(r'\?').astype(int)
#         features['has_exclamation'] = text_series.str.contains(r'\!').astype(int)
#         features['has_number'] = text_series.str.contains(r'\d').astype(int)
#         features['has_emoji'] = text_series.apply(lambda x: int(emoji.emoji_count(x) > 0))
        
#         # Keyword features
#         features['has_power_word'] = text_series.apply(self._contains_power_word)
#         features['clickbait_score'] = text_series.apply(self._calculate_clickbait_score)
        
#         # Topic modeling
#         features['is_movie_related'] = text_series.str.contains(
#             r'movie|film|cinema|actor|actress|director|trailer', 
#             case=False
#         ).astype(int)
        
#         # Tokenization and lemmatization for later use
#         processed_text = text_series.apply(self._tokenize_and_lemmatize)
#         features['processed_text'] = processed_text
        
#         return pd.DataFrame(features)
    
#     def _extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Extract features from other columns"""
#         features = {}
        
#         # Duration features
#         df['duration'] = pd.to_timedelta(df['duration'])
#         features['duration_seconds'] = df['duration'].dt.total_seconds()
        
#         # Category features (one-hot encoded)
#         encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#         category_encoded = encoder.fit_transform(df[['category']])
#         category_df = pd.DataFrame(
#             category_encoded,
#             columns=[f"category_{cat}" for cat in encoder.categories_[0]]
#         )
        
#         # Country features
#         features['is_india'] = (df['country'] == 'India').astype(int)
#         features['is_us'] = (df['country'] == 'United Sta').astype(int)  # Note: Fix data typo
        
#         # Subscriber count (log transformed)
#         features['log_subscribers'] = np.log1p(df['subscriber:'])
        
#         # Combine all metadata features
#         metadata_df = pd.DataFrame(features)
#         return pd.concat([metadata_df, category_df], axis=1)
    
#     def _tokenize_and_lemmatize(self, text: str) -> str:
#         """Tokenize and lemmatize text for later vectorization"""
#         tokens = word_tokenize(text)
#         lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens 
#                      if token not in self.stop_words and token.isalpha()]
#         return ' '.join(lemmatized)
    
#     def _contains_power_word(self, text: str) -> int:
#         """Check if text contains common power words"""
#         power_words = [
#             'secret', 'proven', 'ultimate', 'best', 'top', 'amazing',
#             'incredible', 'essential', 'perfect', 'complete', 'advanced',
#             'exclusive', 'instant', 'quick', 'easy', 'guaranteed'
#         ]
#         return int(any(word in text for word in power_words))
    
#     def _calculate_clickbait_score(self, text: str) -> float:
#         """Calculate a simple clickbait score"""
#         clickbait_phrases = [
#             r'you won\'t believe', r'shocked to see', r'what happens next',
#             r'before you watch', r'the truth about', r'going viral',
#             r'blow your mind', r'can\'t be unseen'
#         ]
#         score = 0
#         for phrase in clickbait_phrases:
#             if re.search(phrase, text, re.IGNORECASE):
#                 score += 1
#         return score / len(clickbait_phrases)  # Normalized score
    
#     def _save_data(self, df: pd.DataFrame):
#         """Save processed data to CSV"""
#         os.makedirs(os.path.dirname(self.config.preprocessed_data_path), exist_ok=True)
#         df.to_csv(self.config.preprocessed_data_path, index=False)
#         logging.info(f"Saved processed text data to {self.config.preprocessed_data_path}")

# if __name__ == "__main__":
#     try:
#         print("Starting text data preprocessing...")
#         preprocessor = TextDataPreprocessor()
#         processed_data = preprocessor.preprocess_text_data()
#         print("Preprocessing completed successfully!")
#         print(processed_data.head())
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         import traceback
#         traceback.print_exc()




import os
import sys
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import emoji
from collections import Counter

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Configure NLTK data path
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data with error handling
def download_nltk_data():
    required_data = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('omw-1.4', 'corpora/omw-1.4'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('punkt_tab', 'tokenizers/punkt_tab')
    ]
    
    for resource, path in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

download_nltk_data()

@dataclass
class TextProcessingConfig:
    preprocessed_data_path: str = os.path.join('artifacts', 'text_features.csv')
    raw_data_path: str = os.path.join('data', 'youtube_dataset_with_full_countries.csv')

class TextDataPreprocessor:
    def __init__(self):
        self.config = TextProcessingConfig()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add YouTube-specific stop words
        self.stop_words.update(['youtube', 'video', 'watch', 'channel', 'subscribe', 'like', 'share'])
        
    def preprocess_text_data(self):
        try:
            logging.info("Starting text data preprocessing")
            
            # Load raw data
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Loaded raw data with {len(df)} rows")
            
            # Basic cleaning
            df = self._clean_data(df)
            
            # Extract text features
            text_features = self._extract_text_features(df['title'])
            
            # Combine with metadata
            metadata_features = self._extract_metadata_features(df)
            final_df = pd.concat([text_features, metadata_features], axis=1)
            
            # Save processed data
            self._save_data(final_df)
            logging.info("Text preprocessing completed successfully")
            
            return final_df
            
        except Exception as e:
            logging.error(f"Error in text preprocessing: {str(e)}")
            raise CustomException(e, sys)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning"""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Drop duplicates
        df = df.drop_duplicates(subset=['title']).copy()
        
        # Handle missing values
        df.loc[:, 'title'] = df['title'].fillna('')
        
        # Clean title text
        df.loc[:, 'title'] = df['title'].apply(self._clean_text)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_text_features(self, text_series: pd.Series) -> pd.DataFrame:
        """Extract NLP features from titles"""
        features = {}
        
        # Basic text features
        features['title_length'] = text_series.apply(len)
        features['word_count'] = text_series.apply(lambda x: len(x.split()))
        features['avg_word_length'] = text_series.apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x else 0
        )
        
        # Sentiment analysis
        features['sentiment_polarity'] = text_series.apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        features['sentiment_subjectivity'] = text_series.apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        
        # Structural features
        features['has_question'] = text_series.str.contains(r'\?').astype(int)
        features['has_exclamation'] = text_series.str.contains(r'\!').astype(int)
        features['has_number'] = text_series.str.contains(r'\d').astype(int)
        features['has_emoji'] = text_series.apply(lambda x: int(emoji.emoji_count(x) > 0))
        
        # Keyword features
        features['has_power_word'] = text_series.apply(self._contains_power_word)
        features['clickbait_score'] = text_series.apply(self._calculate_clickbait_score)
        
        # Topic modeling
        features['is_movie_related'] = text_series.str.contains(
            r'movie|film|cinema|actor|actress|director|trailer', 
            case=False
        ).astype(int)
        
        # Tokenization and lemmatization for later use (with error handling)
        processed_text = text_series.apply(self._safe_tokenize_and_lemmatize)
        features['processed_text'] = processed_text
        
        return pd.DataFrame(features)
    
    def _safe_tokenize_and_lemmatize(self, text: str) -> str:
        """Wrapper with error handling for tokenization"""
        try:
            return self._tokenize_and_lemmatize(text)
        except Exception as e:
            logging.warning(f"Tokenization failed for text: {text[:50]}... Error: {str(e)}")
            return ""
    
    def _extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from other columns"""
        features = {}
        
        # Duration features
        try:
            df['duration'] = pd.to_timedelta(df['duration'])
            features['duration_seconds'] = df['duration'].dt.total_seconds()
        except Exception as e:
            logging.warning(f"Duration processing failed: {str(e)}")
            features['duration_seconds'] = 0
        
        # Category features (one-hot encoded)
        try:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            category_encoded = encoder.fit_transform(df[['category']])
            category_df = pd.DataFrame(
                category_encoded,
                columns=[f"category_{cat}" for cat in encoder.categories_[0]]
            )
        except Exception as e:
            logging.warning(f"Category encoding failed: {str(e)}")
            category_df = pd.DataFrame()
        
        # Country features
        features['is_india'] = (df['country'] == 'India').astype(int)
        features['is_us'] = (df['country'] == 'United Sta').astype(int)
        
        # Subscriber count (log transformed)
        features['log_subscribers'] = np.log1p(df['subscriber:'].fillna(0))
        
        # Combine all metadata features
        metadata_df = pd.DataFrame(features)
        return pd.concat([metadata_df, category_df], axis=1)
    
    def _tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text for later vectorization"""
        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and token.isalpha()]
        return ' '.join(lemmatized)
    
    def _contains_power_word(self, text: str) -> int:
        """Check if text contains common power words"""
        power_words = [
            'secret', 'proven', 'ultimate', 'best', 'top', 'amazing',
            'incredible', 'essential', 'perfect', 'complete', 'advanced',
            'exclusive', 'instant', 'quick', 'easy', 'guaranteed'
        ]
        return int(any(word in text for word in power_words))
    
    def _calculate_clickbait_score(self, text: str) -> float:
        """Calculate a simple clickbait score"""
        clickbait_phrases = [
            r'you won\'t believe', r'shocked to see', r'what happens next',
            r'before you watch', r'the truth about', r'going viral',
            r'blow your mind', r'can\'t be unseen'
        ]
        score = 0
        for phrase in clickbait_phrases:
            if re.search(phrase, text, re.IGNORECASE):
                score += 1
        return score / len(clickbait_phrases)  # Normalized score
    
    def _save_data(self, df: pd.DataFrame):
        """Save processed data to CSV"""
        os.makedirs(os.path.dirname(self.config.preprocessed_data_path), exist_ok=True)
        df.to_csv(self.config.preprocessed_data_path, index=False)
        logging.info(f"Saved processed text data to {self.config.preprocessed_data_path}")

if __name__ == "__main__":
    try:
        print("Starting text data preprocessing...")
        print("Verifying NLTK data...")
        download_nltk_data()  # Ensure all resources are available
        
        print("Initializing preprocessor...")
        preprocessor = TextDataPreprocessor()
        
        print("Processing data...")
        processed_data = preprocessor.preprocess_text_data()
        
        print("\nPreprocessing completed successfully!")
        print("First 5 rows of processed data:")
        print(processed_data.head())
        
    except Exception as e:
        print(f"\nError occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)