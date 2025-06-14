import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageStat
import logging
from pathlib import Path

class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_features(self, image_path: str) -> dict:
        """Extract all required thumbnail features"""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Convert to numpy array for OpenCV processing
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Extract features
            features = {
                'faces': self._detect_faces(img_cv),
                'has_text': self._detect_text(img_cv),
                'brightness': self._calculate_brightness(img),
                'vibrancy': self._calculate_vibrancy(img),
                'contrast': self._calculate_contrast(img_cv),
                'color_variety': self._calculate_color_variety(img),
                'red_dominance': self._calculate_red_dominance(img),
                'sharpness': self._calculate_sharpness(img),
                'aspect_ratio': self._calculate_aspect_ratio(img)
            }
            
            self.logger.info(f"Extracted image features: {features}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise
            
    def _detect_faces(self, img_cv) -> int:
        """Detect number of faces using Haar cascade"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces)
        
    def _detect_text(self, img_cv) -> int:
        """Simple text detection (returns 1 if text likely exists)"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Heuristic: if many small contours, likely has text
        small_contours = [cnt for cnt in contours if 20 < cv2.contourArea(cnt) < 500]
        return 1 if len(small_contours) > 3 else 0
        
    def _calculate_brightness(self, img) -> float:
        """Calculate average brightness (0-255)"""
        stat = ImageStat.Stat(img)
        return sum(stat.mean) / 3  # Average of R,G,B
        
    def _calculate_vibrancy(self, img) -> float:
        """Calculate color vibrancy (standard deviation of colors)"""
        stat = ImageStat.Stat(img)
        return sum(stat.stddev) / 3  # Average of R,G,B std
        
    def _calculate_contrast(self, img_cv) -> float:
        """Calculate contrast using RMS contrast"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return gray.std()
        
    def _calculate_color_variety(self, img) -> float:
        """Calculate color variety (unique colors)"""
        colors = img.getcolors(maxcolors=100000)
        if colors is None:  # Too many colors to count
            return 100
        return sum([count for count, color in colors])
        
    def _calculate_red_dominance(self, img) -> float:
        """Calculate red dominance in image"""
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean
        return (r - (g + b)/2) / 255  # Normalized
        
    def _calculate_sharpness(self, img) -> float:
        """Estimate sharpness using edge detection"""
        img_gray = img.convert('L')
        edges = img_gray.filter(ImageFilter.FIND_EDGES)
        stat = ImageStat.Stat(edges)
        return stat.rms[0]  # Root mean square of edge pixels
        
    def _calculate_aspect_ratio(self, img) -> float:
        """Calculate width/height ratio"""
        width, height = img.size
        return width / height