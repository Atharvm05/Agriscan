#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock version of the AgriScan Flask application for testing without TensorFlow

This is a simplified version of the app.py file that doesn't require TensorFlow
to be installed. It can be used for testing the basic web application functionality.
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory

# Create Flask application
app = Flask(__name__)

# Set base directory
BASE_DIR = Path(__file__).resolve().parent

# Configure app
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Mock disease information
DISEASE_INFO = {
    "Tomato_Healthy": {
        "description": "The tomato plant is healthy and shows no signs of disease.",
        "treatment": "No treatment needed. Continue regular care practices.",
        "prevention": "Maintain good gardening practices including proper watering, adequate spacing, and regular monitoring."
    },
    "Tomato_Early_blight": {
        "description": "Early blight is a fungal disease that affects tomato plants, causing brown spots with concentric rings on leaves.",
        "treatment": "Remove and destroy infected leaves. Apply appropriate fungicides. Ensure good air circulation.",
        "prevention": "Rotate crops, avoid overhead watering, space plants properly, and use disease-resistant varieties."
    },
    "Tomato_Late_blight": {
        "description": "Late blight is a serious fungal disease that causes dark, water-soaked spots on leaves, stems, and fruits.",
        "treatment": "Remove infected plants immediately. Apply copper-based fungicides. Harvest healthy fruits early.",
        "prevention": "Use resistant varieties, avoid overhead watering, ensure good air circulation, and apply preventative fungicides during wet weather."
    },
    "Apple_Scab": {
        "description": "Apple scab is a fungal disease that causes dark, scabby lesions on apple tree leaves and fruit.",
        "treatment": "Remove and destroy fallen leaves. Apply fungicides during the growing season.",
        "prevention": "Plant resistant varieties, ensure good air circulation, and apply preventative fungicides."
    },
    "Apple_Black_rot": {
        "description": "Black rot is a fungal disease affecting apple trees, causing circular lesions on leaves and rotting fruit.",
        "treatment": "Prune infected branches. Remove mummified fruits. Apply appropriate fungicides.",
        "prevention": "Maintain tree health, remove dead wood, and practice good orchard sanitation."
    },
    "Corn_Common_rust": {
        "description": "Common rust is a fungal disease of corn that produces rusty-colored pustules on leaves.",
        "treatment": "Apply fungicides early in the infection. Remove severely infected plants.",
        "prevention": "Plant resistant hybrids, ensure proper spacing, and apply preventative fungicides in high-risk areas."
    },
    "Potato_Early_blight": {
        "description": "Early blight in potatoes causes dark brown spots with concentric rings on lower leaves.",
        "treatment": "Remove infected leaves. Apply appropriate fungicides. Ensure good air circulation.",
        "prevention": "Use certified disease-free seed potatoes, practice crop rotation, and maintain proper plant spacing."
    },
    "Grape_Black_rot": {
        "description": "Black rot is a fungal disease of grapes causing leaf spots and rotting fruit.",
        "treatment": "Remove infected fruit and leaves. Apply fungicides according to recommended schedule.",
        "prevention": "Prune for good air circulation, remove mummified fruit, and apply preventative fungicides."
    },
    "Strawberry_Leaf_scorch": {
        "description": "Leaf scorch in strawberries causes purple to red spots on leaves that eventually dry and turn brown.",
        "treatment": "Remove infected leaves. Apply appropriate fungicides. Improve air circulation.",
        "prevention": "Use disease-free plants, avoid overhead irrigation, and maintain proper plant spacing."
    },
    "Soybean_Bacterial_blight": {
        "description": "Bacterial blight in soybeans causes water-soaked spots on leaves that turn yellow to brown.",
        "treatment": "No effective chemical control once infected. Remove severely infected plants.",
        "prevention": "Use disease-free seeds, practice crop rotation, and avoid working in fields when plants are wet."
    },
    "Banana_Healthy": {
        "description": "The banana plant is healthy and shows no signs of disease.",
        "treatment": "No treatment needed. Continue regular care practices.",
        "prevention": "Maintain good gardening practices including proper watering, adequate spacing, and regular monitoring."
    },
    "Banana_Black_Sigatoka": {
        "description": "Black Sigatoka is a serious fungal disease that causes black streaks and spots on banana leaves, eventually leading to leaf death.",
        "treatment": "Remove infected leaves. Apply appropriate fungicides on a regular schedule. Ensure good drainage.",
        "prevention": "Use resistant varieties, maintain proper spacing, and implement good sanitation practices."
    },
    "Banana_Panama_Disease": {
        "description": "Panama disease (Fusarium wilt) is a devastating fungal disease that causes yellowing and wilting of leaves, eventually killing the plant.",
        "treatment": "No effective treatment once infected. Remove and destroy infected plants.",
        "prevention": "Use disease-free planting material, plant resistant varieties, and avoid introducing the pathogen through contaminated soil or tools."
    }
}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/offline')
def offline():
    return render_template('offline.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('static/js', 'service-worker.js')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file part in the request',
            'predictions': [],
            'inference_time': 'N/A'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'error': 'No selected file',
            'predictions': [],
            'inference_time': 'N/A'
        }), 400
    
    if file:
        # Mock prediction result
        import random
        import time
        import os
        import io
        try:
            from PIL import Image
        except ImportError:
            print("Error: Pillow is required for image processing")
            # Fall back to not using image validation if PIL is not available
            Image = None
        from werkzeug.utils import secure_filename
        
        # Save the file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"  # Add timestamp to avoid filename conflicts
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Check if the image is likely a plant leaf by analyzing colors
        try:
            # Only perform image validation if PIL/Pillow is available
            if Image is not None:
                # Open the image and convert to RGB
                img = Image.open(file_path).convert('RGB')
                
                # Resize for faster processing
                img = img.resize((100, 100))
                pixels = list(img.getdata())
                
                # Count green pixels (simple heuristic for plant detection)
                green_pixels = 0
                total_pixels = len(pixels)
                
                for pixel in pixels:
                    r, g, b = pixel
                    # Check if green is the dominant color in the pixel
                    if g > r and g > b and g > 100:
                        green_pixels += 1
                
                green_ratio = green_pixels / total_pixels
                print(f"Debug: Green ratio in image: {green_ratio:.2f}")
                
                # If less than 15% green, probably not a plant
                if green_ratio < 0.15:
                    return jsonify({
                        'error': 'The image does not appear to be a plant leaf. Please upload an image of a plant leaf for analysis.',
                        'predictions': [],
                        'inference_time': 'N/A'
                    }), 400
                
            # Group diseases by plant type
            plant_groups = {
                'tomato': [key for key in DISEASE_INFO.keys() if key.startswith('Tomato')],
                'apple': [key for key in DISEASE_INFO.keys() if key.startswith('Apple')],
                'corn': [key for key in DISEASE_INFO.keys() if key.startswith('Corn')],
                'potato': [key for key in DISEASE_INFO.keys() if key.startswith('Potato')],
                'grape': [key for key in DISEASE_INFO.keys() if key.startswith('Grape')],
                'strawberry': [key for key in DISEASE_INFO.keys() if key.startswith('Strawberry')],
                'soybean': [key for key in DISEASE_INFO.keys() if key.startswith('Soybean')],
                'banana': [key for key in DISEASE_INFO.keys() if key.startswith('Banana')]
            }
            
            # Determine plant type based on image analysis or random selection
            if Image is not None:
                # Analyze image to determine plant type (simplified color analysis)
                # This is a very basic approach - in a real app, you'd use ML for this
                r_avg = sum(p[0] for p in pixels) / total_pixels
                g_avg = sum(p[1] for p in pixels) / total_pixels
                b_avg = sum(p[2] for p in pixels) / total_pixels
                
                print(f"Debug: Color averages - R: {r_avg:.1f}, G: {g_avg:.1f}, B: {b_avg:.1f}")
                
                # Improved color-based plant type selection with more distinct thresholds
                # Calculate color ratios for better differentiation
                rg_ratio = r_avg / g_avg if g_avg > 0 else 0
                rb_ratio = r_avg / b_avg if b_avg > 0 else 0
                gb_ratio = g_avg / b_avg if b_avg > 0 else 0
                
                print(f"Debug: Color ratios - R/G: {rg_ratio:.2f}, R/B: {rb_ratio:.2f}, G/B: {gb_ratio:.2f}")
                
                # More distinct classification based on color characteristics
                if r_avg > 80 and r_avg < 130 and g_avg > 130 and g_avg < 180 and b_avg < 80 and rg_ratio < 0.85:  # Yellow-green with high green (banana leaf)
                    plant_type = 'banana'
                elif r_avg > 120 and g_avg > 120 and b_avg < 90 and rg_ratio > 0.9 and rg_ratio < 1.1:  # Yellow-green
                    plant_type = 'corn'
                elif r_avg < 90 and g_avg > 110 and b_avg < 90 and rg_ratio < 0.8 and gb_ratio > 1.3:  # Deep green
                    plant_type = 'apple'
                elif r_avg < 70 and g_avg > 100 and b_avg < 70 and rg_ratio < 0.7:  # Dark green
                    plant_type = 'potato'
                elif r_avg > 100 and g_avg > 130 and b_avg > 80 and gb_ratio > 1.2 and gb_ratio < 1.8:  # Bright green
                    plant_type = 'tomato'
                elif r_avg > 90 and g_avg > 100 and b_avg > 90 and rg_ratio < 0.95 and rb_ratio < 1.1:  # Light green with blue tint
                    plant_type = 'grape'
                elif r_avg > 100 and g_avg > 100 and b_avg < 80 and rg_ratio > 0.95 and rg_ratio < 1.05:  # Balanced yellow-green
                    plant_type = 'soybean'
                elif r_avg > 80 and r_avg < 120 and g_avg > 100 and g_avg < 140 and b_avg < 100:  # Medium green
                    plant_type = 'strawberry'
                else:  # Use deterministic selection if no clear match
                    # Create a deterministic seed based on image characteristics
                    if 'file' in request.files and request.files['file'].filename:
                        filename = secure_filename(request.files['file'].filename)
                        filename_seed = sum(ord(c) for c in filename)
                        img_seed = int((r_avg * 100) + (g_avg * 10) + b_avg)
                        combined_seed = img_seed + filename_seed
                        # Set seed for deterministic selection
                        random.seed(combined_seed)
                        print(f"Debug: Using deterministic seed {combined_seed} for plant type selection")
                        # Get sorted list of plant types for consistent selection
                        plant_types = sorted(list(plant_groups.keys()))
                        # Select plant type based on seed
                        index = combined_seed % len(plant_types)
                        plant_type = plant_types[index]
                        # Reset random seed
                        random.seed(None)
                    else:
                        # Fallback to a default plant type if no file
                        plant_type = 'tomato'
            else:
                # If PIL is not available, use a default plant type instead of random selection
                plant_type = 'tomato'  # Default to tomato as the most common plant type
                print(f"Debug: PIL not available, using default plant type: {plant_type}")
            
            print(f"Debug: Selected plant type: {plant_type}")
            
            # Get diseases for the selected plant type
            available_diseases = plant_groups.get(plant_type, [])
            
            # If no diseases found for this plant type, use a deterministic fallback
            if not available_diseases:
                # Use a deterministic fallback to tomato (which always has diseases defined)
                plant_type = 'tomato'
                available_diseases = plant_groups.get(plant_type, [])
                print(f"Debug: No diseases found for original plant type, using fallback: {plant_type}")
            
            # Generate predictions
            top_predictions = []
            
            # Make a copy of available diseases to avoid modifying the original
            disease_options = available_diseases.copy()
            
            # Use image characteristics to create a deterministic seed for disease selection
            if Image is not None and 'file' in request.files and request.files['file'].filename:
                # Create a deterministic seed based on image characteristics
                img_seed = int((r_avg * 100) + (g_avg * 10) + b_avg)
                # Use filename as additional seed component for consistency
                filename = secure_filename(request.files['file'].filename)
                filename_seed = sum(ord(c) for c in filename)
                # Combine seeds
                combined_seed = img_seed + filename_seed
                # Set the random seed for deterministic selection
                random.seed(combined_seed)
                print(f"Debug: Using deterministic seed {combined_seed} for disease prediction")
            
            # Generate up to 3 predictions (or fewer if not enough diseases available)
            for i in range(min(3, len(disease_options))):
                # Select a disease deterministically based on the seed
                # Sort the options to ensure consistent selection order
                sorted_options = sorted(disease_options)
                # Use a deterministic selection method
                selection_index = i % len(sorted_options)
                predicted_class = sorted_options[selection_index]
                # Remove it from the list to avoid duplicates
                disease_options.remove(predicted_class)
                
                # Generate deterministic confidence based on index
                if i == 0:
                    confidence = 0.85 + ((combined_seed % 15) / 100) if 'combined_seed' in locals() else 0.9
                else:
                    confidence = 0.4 + ((combined_seed % 30) / 100) if 'combined_seed' in locals() else 0.5
                
                # Get disease information
                disease_info = DISEASE_INFO.get(predicted_class, {
                    "description": "Information not available",
                    "treatment": "Information not available",
                    "prevention": "Information not available"
                })
                
                # Format the class name for display
                display_name = predicted_class.replace('_', ' ')
                
                prediction = {
                    'rank': i + 1,
                    'class_id': i,
                    'class_name': predicted_class,
                    'display_name': display_name,
                    'description': disease_info['description'],
                    'confidence': float(confidence),
                    'confidence_percent': f"{confidence * 100:.2f}%",
                    'treatment': disease_info['treatment'],
                    'prevention': disease_info['prevention']
                }
                
                top_predictions.append(prediction)
            
            # Reset random seed after disease prediction to avoid affecting other operations
            if 'combined_seed' in locals():
                # Use system time to reset the random seed
                random.seed(None)
                print("Debug: Reset random seed after disease prediction")
            
        except Exception as e:
            # If image processing fails, fall back to random predictions
            print(f"Error analyzing image: {str(e)}")
            diseases = list(DISEASE_INFO.keys())
            top_predictions = []
            
            # Generate 3 random predictions
            for i in range(3):
                # Select a random disease
                predicted_class = random.choice(diseases)
                # Remove it from the list to avoid duplicates
                diseases.remove(predicted_class)
                
                # Generate a random confidence (higher for first prediction)
                if i == 0:
                    confidence = random.uniform(0.7, 0.99)
                else:
                    confidence = random.uniform(0.3, 0.7)
                
                # Get disease information
                disease_info = DISEASE_INFO.get(predicted_class, {
                    "description": "Information not available",
                    "treatment": "Information not available",
                    "prevention": "Information not available"
                })
                
                # Format the class name for display
                display_name = predicted_class.replace('_', ' ')
                
                prediction = {
                    'rank': i + 1,
                    'class_id': i,
                    'class_name': predicted_class,
                    'display_name': display_name,
                    'description': disease_info['description'],
                    'confidence': float(confidence),
                    'confidence_percent': f"{confidence * 100:.2f}%",
                    'treatment': disease_info['treatment'],
                    'prevention': disease_info['prevention']
                }
                
                top_predictions.append(prediction)
        
        # Create result structure matching the real app
        result = {
            'predictions': top_predictions,
            'inference_time': f"{random.uniform(50, 200):.2f} ms",
            'image_path': os.path.join('uploads', filename)
        }
        
        return jsonify(result)
    
    return jsonify({
        'error': 'Failed to process file',
        'predictions': [],
        'inference_time': 'N/A'
    }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('offline.html'), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')