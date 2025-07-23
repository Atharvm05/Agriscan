import os
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path
import json
import shutil

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'model'
TFLITE_MODEL_DIR = MODEL_DIR / 'tflite'
TFJS_MODEL_DIR = MODEL_DIR / 'tfjs'
WEBAPP_MODELS_DIR = BASE_DIR / 'webapp' / 'static' / 'models'

# Create model directories if they don't exist
TFLITE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TFJS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
WEBAPP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Function to find and load the trained model
def load_trained_model():
    # Check if the trained model exists
    model_path = MODEL_DIR / 'agriscan_model.h5'
    if not model_path.exists():
        model_path = MODEL_DIR / 'best_fine_tuned_model.h5'
        if not model_path.exists():
            model_path = MODEL_DIR / 'best_model.h5'
            if not model_path.exists():
                raise FileNotFoundError("No trained model found. Please train the model first.")
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    return model, model_path

# Function to convert model to TensorFlow Lite
def convert_to_tflite():
    print("Converting model to TensorFlow Lite format...")
    
    # Load the trained model
    model, model_path = load_trained_model()
    
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_model_path = TFLITE_MODEL_DIR / 'agriscan_model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to {tflite_model_path}")
    
    # Create a quantized model for even smaller size
    print("Creating quantized model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert the model
    tflite_quantized_model = converter.convert()
    
    # Save the quantized TFLite model
    tflite_quantized_model_path = TFLITE_MODEL_DIR / 'agriscan_model_quantized.tflite'
    with open(tflite_quantized_model_path, 'wb') as f:
        f.write(tflite_quantized_model)
    
    print(f"Quantized TensorFlow Lite model saved to {tflite_quantized_model_path}")
    
    # Get model size information
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)  # Size in MB
    tflite_quantized_size = os.path.getsize(tflite_quantized_model_path) / (1024 * 1024)  # Size in MB
    
    print(f"\nModel Size Comparison:")
    print(f"Original model: {original_size:.2f} MB")
    print(f"TFLite model: {tflite_size:.2f} MB")
    print(f"TFLite quantized model: {tflite_quantized_size:.2f} MB")
    
    return tflite_model_path, tflite_quantized_model_path

# Function to create metadata for the TFLite model
def create_metadata():
    print("\nCreating metadata for TensorFlow Lite model...")
    
    # Load class labels
    class_labels_path = MODEL_DIR / 'class_labels.txt'
    if not class_labels_path.exists():
        print("Class labels file not found. Skipping metadata creation.")
        return
    
    with open(class_labels_path, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]
    
    # Create a dictionary mapping class indices to labels and descriptions
    label_map = {}
    for i, label in enumerate(class_labels):
        # Convert label like 'Tomato_Early_blight' to 'Tomato Early Blight'
        display_name = label.replace('_', ' ').title()
        
        # Create a simple description
        if 'healthy' in label.lower():
            description = f"Healthy {display_name.split()[0]} plant."
        else:
            disease = ' '.join(display_name.split()[1:])
            description = f"{disease} disease in {display_name.split()[0]} plant."
        
        label_map[i] = {
            'name': label,
            'display_name': display_name,
            'description': description
        }
    
    # Save the label map as JSON
    label_map_path = TFLITE_MODEL_DIR / 'label_map.json'
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"Label map saved to {label_map_path}")
    
    # Create a metadata.json file with model information
    metadata = {
        'model_name': 'AgriScan Plant Disease Detection',
        'model_description': 'A model trained to detect diseases in plant leaves',
        'input_shape': [1, 224, 224, 3],  # Batch size, height, width, channels
        'input_type': 'float32',
        'input_normalization': {
            'mean': [0.485, 0.456, 0.406],  # ImageNet means
            'std': [0.229, 0.224, 0.225]   # ImageNet stds
        },
        'num_classes': len(class_labels),
        'classes': class_labels,
        'label_map_path': 'label_map.json'
    }
    
    metadata_path = TFLITE_MODEL_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")

# Function to convert model to TensorFlow.js format
def convert_to_tfjs():
    print("\nConverting model to TensorFlow.js format...")
    
    # Load the trained model
    model, model_path = load_trained_model()
    
    # Convert the model to TensorFlow.js format
    print(f"Converting model to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, TFJS_MODEL_DIR)
    
    print(f"TensorFlow.js model saved to {TFJS_MODEL_DIR}")
    
    # Copy the model to the webapp directory for serving
    webapp_tfjs_dir = WEBAPP_MODELS_DIR / 'tfjs'
    webapp_tfjs_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from TFJS_MODEL_DIR to webapp_tfjs_dir
    for file_path in TFJS_MODEL_DIR.glob('*'):
        if file_path.is_file():
            shutil.copy(file_path, webapp_tfjs_dir / file_path.name)
    
    print(f"TensorFlow.js model copied to {webapp_tfjs_dir} for web serving")
    
    # Get model size information
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    tfjs_size = sum(os.path.getsize(f) for f in TFJS_MODEL_DIR.glob('*') if f.is_file()) / (1024 * 1024)  # Size in MB
    
    print(f"TensorFlow.js model size: {tfjs_size:.2f} MB (Original: {original_size:.2f} MB)")
    
    return TFJS_MODEL_DIR

# Function to test the TFLite model
def test_tflite_model(tflite_model_path):
    print("\nTesting TensorFlow Lite model...")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model details
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Create a dummy input
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Model successfully ran inference on dummy input.")
    print(f"Output shape: {output_data.shape}")

# Function to copy metadata and label files to webapp directory
def copy_files_to_webapp():
    print("\nCopying metadata and label files to webapp directory...")
    
    # Copy label_map.json to webapp directory
    label_map_path = TFLITE_MODEL_DIR / 'label_map.json'
    if label_map_path.exists():
        webapp_label_map_path = WEBAPP_MODELS_DIR / 'label_map.json'
        shutil.copy(label_map_path, webapp_label_map_path)
        print(f"Copied label map to {webapp_label_map_path}")
    
    # Copy metadata.json to webapp directory
    metadata_path = TFLITE_MODEL_DIR / 'metadata.json'
    if metadata_path.exists():
        webapp_metadata_path = WEBAPP_MODELS_DIR / 'metadata.json'
        shutil.copy(metadata_path, webapp_metadata_path)
        print(f"Copied metadata to {webapp_metadata_path}")
    
    # Copy TFLite model to webapp directory
    tflite_model_path = TFLITE_MODEL_DIR / 'agriscan_model.tflite'
    if tflite_model_path.exists():
        webapp_tflite_model_path = WEBAPP_MODELS_DIR / 'agriscan_model.tflite'
        shutil.copy(tflite_model_path, webapp_tflite_model_path)
        print(f"Copied TFLite model to {webapp_tflite_model_path}")
    
    # Copy quantized TFLite model to webapp directory
    tflite_quantized_model_path = TFLITE_MODEL_DIR / 'agriscan_model_quantized.tflite'
    if tflite_quantized_model_path.exists():
        webapp_tflite_quantized_model_path = WEBAPP_MODELS_DIR / 'agriscan_model_quantized.tflite'
        shutil.copy(tflite_quantized_model_path, webapp_tflite_quantized_model_path)
        print(f"Copied quantized TFLite model to {webapp_tflite_quantized_model_path}")

# Main function
def main():
    print("Starting model conversion process...")
    
    # Convert model to TFLite
    print("\n=== TensorFlow Lite Conversion ===")
    tflite_model_path, tflite_quantized_model_path = convert_to_tflite()
    
    # Create metadata
    create_metadata()
    
    # Test the TFLite model
    test_tflite_model(tflite_model_path)
    
    # Convert model to TensorFlow.js
    print("\n=== TensorFlow.js Conversion ===")
    tfjs_model_dir = convert_to_tfjs()
    
    # Copy files to webapp directory
    copy_files_to_webapp()
    
    print("\nModel conversion completed successfully!")
    print(f"TFLite models saved to {TFLITE_MODEL_DIR}")
    print(f"TensorFlow.js models saved to {TFJS_MODEL_DIR}")
    print(f"Models copied to webapp directory: {WEBAPP_MODELS_DIR}")
    print("\nYou can now use these models in the web application for offline inference.")

if __name__ == "__main__":
    main()