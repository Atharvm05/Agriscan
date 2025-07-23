import os
import numpy as np
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path
import time
import random

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'model'
TFLITE_MODEL_DIR = MODEL_DIR / 'tflite'
UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / 'webapp' / 'static' / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize Flask app
app = Flask(__name__, 
           static_folder=str(Path(__file__).resolve().parent.parent / 'webapp' / 'static'),
           template_folder=str(Path(__file__).resolve().parent.parent / 'webapp' / 'templates'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load label map
def load_label_map():
    # Check if label map exists
    if (TFLITE_MODEL_DIR / 'label_map.json').exists():
        with open(TFLITE_MODEL_DIR / 'label_map.json', 'r') as f:
            return json.load(f)
    
    # If no labels found, return empty map
    return {}

# Load disease information
def load_disease_info():
    disease_info_path = Path(__file__).resolve().parent.parent / 'webapp' / 'static' / 'data' / 'disease_info.json'
    
    # Check if disease info file exists
    if disease_info_path.exists():
        with open(disease_info_path, 'r') as f:
            return json.load(f)
    
    # If disease info file doesn't exist, return empty dict
    return {}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Mock inference function for Vercel deployment
def mock_inference(image_path):
    # Load label map
    label_map = load_label_map()
    
    # If label map is empty, return error
    if not label_map:
        return {'error': 'Label map not found'}
    
    # Get random class index
    class_idx = random.randint(0, len(label_map) - 1)
    
    # Get class name
    class_name = label_map[str(class_idx)]
    
    # Create mock prediction
    predictions = {}
    for i in range(len(label_map)):
        if i == class_idx:
            predictions[label_map[str(i)]] = random.uniform(0.7, 0.95)
        else:
            predictions[label_map[str(i)]] = random.uniform(0.0, 0.1)
    
    # Sort predictions
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 3 predictions
    return {
        'class_name': class_name,
        'predictions': sorted_predictions[:3]
    }

# Routes
@app.route('/')
def root():
    return redirect('/index')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform inference
        try:
            result = mock_inference(filepath)
            
            # Add image path to result
            result['image_path'] = f"/static/uploads/{filename}"
            
            # Add disease info to result
            disease_info = load_disease_info()
            if result['class_name'] in disease_info:
                result['disease_info'] = disease_info[result['class_name']]
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(app.static_folder, 'manifest.json')

@app.route('/service-worker.js')
def serve_service_worker():
    return send_from_directory(app.static_folder, 'js/service-worker.js')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))