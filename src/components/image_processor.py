# import os
# import sys
# from src.exception import CustomException
# from src.logger  import logging
# import pandas as pd
# import cv2
# from dataclasses import dataclass
# import os
# from PIL import Image
# from pathlib import Path
# import numpy as np


# project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(project_root)



# class YoutubeThumbnailAnalyzer:
#     def __init__(self):
#         self.facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         print(f"Cascade classifier loaded: {not self.facecascade.empty()}")
#         if self.facecascade.empty():
#          raise ValueError("Failed to load Haar cascade classifier")
#         self.text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         logging.info("thumbnail analyzer started with models")

#     def analyze_directory(self,thumbnail_dir:str,output_csv:str)->pd.DataFrame:
#         try:
#             logging.info(f'startint thumbanil analysis from directory {thumbnail_dir}')    


#             # get all image files from directory that we has given in arguiments

#             image_files=self._get_image_files(thumbnail_dir)
#             if not image_files:
#                 raise ValueError("no vaild images found")
            
#             # process all image susing functions below
#             features=[]

#             for img_file in image_files:
#                 img_path=os.path.join(thumbnail_dir,img_file)
#                 try:
#                     img_features=self._analyze_single_image(img_path)
#                     img_features['image_name']=img_file
#                     features.append(img_features)

#                 except Exception as e:
#                     raise CustomException(e,sys)
#                     logging.info(f"failed to process image :{img_file}")  
#                     continue  

#             features_df=pd.DataFrame(features)
#             self._save_features(features_df,output_csv)    

#             logging.info(f"succesfully processed {len(features)}thumbnails")
#             return features_df
        

#         except Exception as e:
#             logging.info("thumbnail analysis fsailed ")
#             raise CustomException(e,sys)
        

#     def _get_image_files(self,directory:str)->list:
#         valid_extensions=('.jpg','.jpeg','.png')
#         return [
#             f for f in os.listdir(directory)
#             if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(directory,f))
#         ] 
    
#     def _analyze_single_image(self,image_path:str) ->dict:
#         img_cv=cv2.imread(image_path)
#         if img_cv is None:
#             raise ValueError(f"could not read image :{image_path}")
#         img_pil=Image.open(image_path)


#         return {
#             'faces': self._count_faces(img_cv),
#             'has_text': self._detect_text(img_cv),
#             'brightness': round(self._get_brightness(img_cv), 4),
#             'vibrancy': round(self._get_vibrancy(img_cv), 4),
#             'contrast': round(self._get_contrast(img_cv), 4),
#             'color_variety': round(self._get_color_variety(img_pil), 4),
#             'red_dominance': round(self._get_red_dominance(img_cv), 4),
#             'sharpness': round(self._get_sharpness(img_cv), 4),
#             'aspect_ratio': round(self._get_aspect_ratio(img_cv), 2)
#         }
    


#     def _save_features(self,df:pd.DataFrame,output_path:str):
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         df.to_csv(output_path, index=False)
#         logging.info(f"Saved features to: {output_path}")



#     def _count_faces(self, img) -> int:
    
#       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#       faces = self.facecascade.detectMultiScale(gray, 1.1, 4)
#       print(f"Type of faces: {type(faces)}, Value: {faces}")  # Debug print
#       if faces is None:
#          return 0
#       return len(faces)
    


#     def _detect_text(self, img) -> int:
#         """Detect presence of text (1/0)"""
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)
#         edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
#         return 1 if edge_density > 0.03 else 0

#     def _get_brightness(self, img) -> float:
#         """Calculate normalized brightness (0-1)"""
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         return np.mean(hsv[:,:,2]) / 255

#     def _get_vibrancy(self, img) -> float:
#         """Calculate color saturation (0-1)"""
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         return np.mean(hsv[:,:,1]) / 255

#     def _get_contrast(self, img) -> float:
#         """Calculate contrast using RMS"""
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return np.std(gray) / 255

#     def _get_color_variety(self, img) -> float:
#         """Count unique colors (normalized)"""
#         img = img.convert('RGB')
#         colors = img.getcolors(maxcolors=100000)
#         return len(colors) / 1000  # Normalized

#     def _get_red_dominance(self, img) -> float:
#         """Calculate percentage of red pixels"""
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         lower_red = np.array([0, 70, 50])
#         upper_red = np.array([10, 255, 255])
#         mask = cv2.inRange(hsv, lower_red, upper_red)
#         return np.sum(mask > 0) / (img.shape[0] * img.shape[1])

#     def _get_sharpness(self, img) -> float:
#         """Measure image sharpness"""
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return cv2.Laplacian(gray, cv2.CV_64F).var()

#     def _get_aspect_ratio(self, img) -> float:
#         """Calculate width/height ratio"""
#         return img.shape[1] / img.shape[0]  # width/height

# if __name__ == "__main__":
#     # Example usage
#     try:
#         analyzer = YoutubeThumbnailAnalyzer()
#         features_df = analyzer.analyze_directory(
#             thumbnail_dir="data/thumbnails",
#             output_csv="artifacts/thumbnail_features.csv"
#         )
#         print(features_df.head())
#     except Exception as e:
#         print(f"Error: {str(e)}")    





import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import cv2
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
import numpy as np

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

class YoutubeThumbnailAnalyzer:
    def __init__(self):
        self.facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print(f"Cascade classifier loaded: {not self.facecascade.empty()}")
        if self.facecascade.empty():
            raise ValueError("Failed to load Haar cascade classifier")
        self.text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        logging.info("thumbnail analyzer started with models")

    def analyze_directory(self, thumbnail_dir: str, output_csv: str) -> pd.DataFrame:
        try:
            logging.info(f'starting thumbnail analysis from directory {thumbnail_dir}')

            # Get all image files from directory
            image_files = self._get_image_files(thumbnail_dir)
            print(f"Found image files: {image_files}")
            if image_files is None:  # Safeguard
                raise ValueError("Image files list is None")
            if not image_files:
                raise ValueError("no valid images found")

            # Process all images using functions below
            features = []

            for img_file in image_files:
                img_path = os.path.join(thumbnail_dir, img_file)
                try:
                    img_features = self._analyze_single_image(img_path)
                    img_features['image_name'] = img_file
                    features.append(img_features)
                except Exception as e:
                    logging.info(f"failed to process image: {img_file}")
                    raise CustomException(e, sys)

            features_df = pd.DataFrame(features)
            self._save_features(features_df, output_csv)

            logging.info(f"successfully processed {len(features)} thumbnails")
            return features_df

        except Exception as e:
            logging.info(f"thumbnail analysis failed: {str(e)}")
            raise CustomException(e, sys)

    def _get_image_files(self, directory: str) -> list:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        valid_extensions = ('.jpg', '.jpeg', '.png')
        files = os.listdir(directory)
        print(f"Files in directory {directory}: {files}")  # Debug print
        image_files = [
            f for f in files
            if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(directory, f))
        ]
        print(f"Filtered image files: {image_files}")  # Debug print
        return image_files

    def _analyze_single_image(self, image_path: str) -> dict:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError(f"could not read image: {image_path}")
        img_pil = Image.open(image_path)

        return {
            'faces': self._count_faces(img_cv),
            'has_text': self._detect_text(img_cv),
            'brightness': round(self._get_brightness(img_cv), 4),
            'vibrancy': round(self._get_vibrancy(img_cv), 4),
            'contrast': round(self._get_contrast(img_cv), 4),
            'color_variety': round(self._get_color_variety(img_pil), 4),
            'red_dominance': round(self._get_red_dominance(img_cv), 4),
            'sharpness': round(self._get_sharpness(img_cv), 4),
            'aspect_ratio': round(self._get_aspect_ratio(img_cv), 2)
        }

    def _save_features(self, df: pd.DataFrame, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved features to: {output_path}")

    def _count_faces(self, img) -> int:
        """Count number of human faces"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.facecascade.detectMultiScale(gray, 1.1, 4)
        print(f"Type of faces: {type(faces)}, Value: {faces}")
        if faces is None:
            return 0
        return len(faces)

    def _detect_text(self, img) -> int:
        """Detect presence of text (1/0)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        return 1 if edge_density > 0.03 else 0

    def _get_brightness(self, img) -> float:
        """Calculate normalized brightness (0-1)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 2]) / 255

    def _get_vibrancy(self, img) -> float:
        """Calculate color saturation (0-1)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 1]) / 255

    def _get_contrast(self, img) -> float:
        """Calculate contrast using RMS"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.std(gray) / 255

    def _get_color_variety(self, img) -> float:
        """Count unique colors (normalized)"""
        img = img.convert('RGB')
        colors = img.getcolors(maxcolors=100000)
        if colors is None:  # Handle case where getcolors returns None
            return 0
        return len(colors) / 1000  # Normalized

    def _get_red_dominance(self, img) -> float:
        """Calculate percentage of red pixels"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return np.sum(mask > 0) / (img.shape[0] * img.shape[1])

    def _get_sharpness(self, img) -> float:
        """Measure image sharpness"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _get_aspect_ratio(self, img) -> float:
        """Calculate width/height ratio"""
        return img.shape[1] / img.shape[0]  # width/height

if __name__ == "__main__":
    print("Starting script execution...")
    try:
        analyzer = YoutubeThumbnailAnalyzer()
        features_df = analyzer.analyze_directory(
            thumbnail_dir="data/thumbnails",
            output_csv="artifacts/thumbnail_features.csv"
        )
        print(features_df.head())
    except Exception as e:
        print(f"Error: {str(e)}")