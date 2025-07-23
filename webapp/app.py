import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import json
from pathlib import Path
import time

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'model'
TFLITE_MODEL_DIR = MODEL_DIR / 'tflite'
UPLOAD_FOLDER = Path(__file__).resolve().parent / 'static' / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load TFLite model
def load_tflite_model():
    # Check if quantized model exists, otherwise use regular model
    if (TFLITE_MODEL_DIR / 'agriscan_model_quantized.tflite').exists():
        model_path = TFLITE_MODEL_DIR / 'agriscan_model_quantized.tflite'
    elif (TFLITE_MODEL_DIR / 'agriscan_model.tflite').exists():
        model_path = TFLITE_MODEL_DIR / 'agriscan_model.tflite'
    else:
        raise FileNotFoundError("No TFLite model found. Please convert the model first.")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    return interpreter

# Load label map
def load_label_map():
    # Check if label map exists
    if (TFLITE_MODEL_DIR / 'label_map.json').exists():
        with open(TFLITE_MODEL_DIR / 'label_map.json', 'r') as f:
            return json.load(f)
    
    # If label map doesn't exist, check if class labels exist
    if (MODEL_DIR / 'class_labels.txt').exists():
        with open(MODEL_DIR / 'class_labels.txt', 'r') as f:
            class_labels = [line.strip() for line in f.readlines()]
        
        # Create a simple label map
        label_map = {}
        for i, label in enumerate(class_labels):
            display_name = label.replace('_', ' ').title()
            label_map[str(i)] = {
                'name': label,
                'display_name': display_name,
                'description': display_name
            }
        
        return label_map
    
    # If no labels found, return empty map
    return {}

# Load disease treatments information
def load_treatments():
    treatments_path = Path(__file__).resolve().parent / 'static' / 'data' / 'treatments.json'
    
    # Check if treatments file exists
    if treatments_path.exists():
        with open(treatments_path, 'r') as f:
            return json.load(f)
    
    # If treatments file doesn't exist, create a default one
    default_treatments = {
        "default": {
            "treatment": "Consult with a local agricultural expert for specific treatment advice.",
            "prevention": "Practice crop rotation, ensure proper spacing for air circulation, and maintain field hygiene."
        },
        "healthy": {
            "treatment": "No treatment needed. The plant appears healthy.",
            "prevention": "Continue good agricultural practices such as proper watering, fertilization, and pest monitoring."
        },
        "bacterial": {
            "treatment": "Remove and destroy infected plants. Apply copper-based bactericides.",
            "prevention": "Use disease-free seeds, practice crop rotation, and avoid overhead irrigation."
        },
        "viral": {
            "treatment": "Remove and destroy infected plants. Control insect vectors with appropriate insecticides.",
            "prevention": "Use virus-free seeds, control weeds that may harbor viruses, and control insect vectors."
        },
        "fungal": {
            "treatment": "Apply appropriate fungicides. Remove and destroy severely infected plant parts.",
            "prevention": "Ensure proper spacing for air circulation, avoid overhead irrigation, and practice crop rotation."
        }
    }
    
    # Create the directory if it doesn't exist
    treatments_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the default treatments
    with open(treatments_path, 'w') as f:
        json.dump(default_treatments, f, indent=2)
    
    return default_treatments

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image for model input
def preprocess_image(image_path, input_size=224):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_size, input_size))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

# Function to get disease category
def get_disease_category(disease_name):
    disease_name = disease_name.lower()
    
    if 'healthy' in disease_name:
        return 'healthy'
    elif any(term in disease_name for term in ['bacterial', 'bacteria']):
        return 'bacterial'
    elif any(term in disease_name for term in ['virus', 'viral', 'mosaic']):
        return 'viral'
    elif any(term in disease_name for term in ['fungal', 'fungi', 'mold', 'rust', 'blight', 'mildew', 'spot']):
        return 'fungal'
    else:
        return 'default'

# Function to predict disease from image
def predict_disease(image_path, interpreter, label_map, treatments):
    # Preprocess the image
    input_data = preprocess_image(image_path)
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get top 3 predictions
    top_indices = np.argsort(output_data[0])[-3:][::-1]
    top_predictions = []
    
    for i, idx in enumerate(top_indices):
        confidence = float(output_data[0][idx])
        label_info = label_map.get(str(idx), {'name': f'Unknown_{idx}', 'display_name': f'Unknown {idx}', 'description': 'Unknown disease'})
        
        # Get disease category and treatment
        disease_category = get_disease_category(label_info['name'])
        treatment_info = treatments.get(disease_category, treatments['default'])
        
        prediction = {
            'rank': i + 1,
            'class_id': int(idx),
            'class_name': label_info['name'],
            'display_name': label_info['display_name'],
            'description': label_info.get('description', label_info['display_name']),
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.2f}%",
            'treatment': treatment_info['treatment'],
            'prevention': treatment_info['prevention']
        }
        
        top_predictions.append(prediction)
    
    result = {
        'predictions': top_predictions,
        'inference_time': f"{inference_time * 1000:.2f} ms"
    }
    
    return result

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file part in the request',
            'predictions': [],
            'inference_time': 'N/A'
        }), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({
            'error': 'No selected file',
            'predictions': [],
            'inference_time': 'N/A'
        }), 400
    
    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'File type not allowed. Please upload a PNG, JPG, or JPEG image.',
            'predictions': [],
            'inference_time': 'N/A'
        }), 400
    
    # Check if model is loaded
    if interpreter is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure the model files are available.',
            'predictions': [],
            'inference_time': 'N/A'
        }), 500
    
    try:
        # Save the file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"  # Add timestamp to avoid filename conflicts
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction
        result = predict_disease(file_path, interpreter, label_map, treatments)
        
        # Add file path to result
        result['image_path'] = os.path.join('uploads', filename)
        
        # Verify result structure
        if 'predictions' not in result or not result['predictions']:
            return jsonify({
                'error': 'No predictions generated for this image',
                'predictions': [],
                'inference_time': result.get('inference_time', 'N/A'),
                'image_path': result.get('image_path')
            }), 500
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        # Return a structured error response with empty predictions array
        return jsonify({
            'error': f"Error processing image: {str(e)}",
            'predictions': [],
            'inference_time': 'N/A'
        }), 500

@app.route('/offline')
def offline():
    return render_template('offline.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('static', 'service-worker.js')

# Load model and label map at startup
print("Loading TensorFlow Lite model...")
try:
    interpreter = load_tflite_model()
    label_map = load_label_map()
    treatments = load_treatments()
    print("Model loaded successfully.")
    print(f"Number of classes: {len(label_map)}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("The application will start, but predictions will not work until a model is available.")
    interpreter = None
    label_map = {}
    treatments = load_treatments()

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)