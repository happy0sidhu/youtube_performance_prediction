from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.components.image_processor import ImageProcessor
import os
from werkzeug.utils import secure_filename
import logging
from src.utils.logger import logger
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

predictor = PredictionPipeline()
image_processor = ImageProcessor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Process form data
            upload_time = datetime.strptime(request.form['upload_time'], '%Y-%m-%dT%H:%M')
            upload_hour = upload_time.hour
            upload_dayofweek = upload_time.weekday()
            
            # Process thumbnail image
            if 'thumbnail' not in request.files:
                return render_template('error.html', error_message="No thumbnail provided"), 400
                
            file = request.files['thumbnail']
            if file.filename == '':
                return render_template('error.html', error_message="No thumbnail selected"), 400
                
            if not (file and allowed_file(file.filename)):
                return render_template('error.html', error_message="Invalid file type"), 400
                
            filename = secure_filename(file.filename)
            thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(thumbnail_path)
            
            # Extract image features
            image_features = image_processor.extract_features(thumbnail_path)
            
            # Prepare input data
            input_data = {
                'title': request.form['title'],
                'duration_seconds': float(request.form['duration']),
                'upload_hour': upload_hour,
                'upload_dayofweek': upload_dayofweek,
                'subscribers': float(request.form['subscribers']),
                'genre': request.form['genre'],
                'country': request.form['country'],
                **image_features
            }
            
            # Make prediction
            prediction = predictor.predict(input_data)
            
            # Format results
            result = {
                'median_views': f"{prediction['median_views']:,}",
                'prediction_range': {
                    'lower': f"{prediction['prediction_range']['lower']:,}",
                    'upper': f"{prediction['prediction_range']['upper']:,}",
                    'confidence': prediction['prediction_range']['confidence']
                },
                'thumbnail_path': thumbnail_path,
                'in_target_range': prediction['is_in_target_range']
            }
            
            return render_template('prediction.html', result=result)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return render_template('error.html', error_message=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)