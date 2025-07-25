# Enhanced Mock TFLite Model for AgriScan

This directory contains an enhanced mock TFLite model for the AgriScan crop disease detection application. The mock model is designed to provide realistic predictions based on image characteristics, making it suitable for testing and demonstration purposes.

## Features

### Enhanced Color Profile Detection
The model analyzes RGB color distributions to identify plant types and disease patterns:
- Calculates average color values and standard deviations
- Computes color ratios for better plant type identification
- Matches against known color profiles for different plant types (Apple, Tomato, Corn, Potato)

### Advanced Texture Analysis
The model performs sophisticated texture analysis to detect disease-specific patterns:
- Calculates higher-order statistical moments (skewness) for detailed texture characterization
- Detects specific disease patterns:
  - **Spots**: High contrast with asymmetric color distribution
  - **Mosaic**: Irregular color distributions with high green channel variation
  - **Blight**: Spreading dark lesions with negative skewness
  - **Rust**: Rusty patches with positive red skewness

### Disease-Specific Feature Detection
The model incorporates knowledge of disease-specific visual characteristics:
- Maps color patterns to known disease presentations
- Applies disease-specific boosting based on visual similarity
- Combines plant type and disease pattern detection for more accurate predictions

### Optimized for Mobile Deployment
- Quantized model available for reduced size and faster inference
- Compatible with TensorFlow Lite for mobile deployment
- Designed for efficient inference on resource-constrained devices

## Files

- `mock_model.py`: Script to generate the mock TFLite models
- `agriscan_model.tflite`: Standard TFLite model
- `agriscan_model_quantized.tflite`: Quantized TFLite model for better performance
- `label_map.json`: Mapping of numerical labels to disease names

## Model Architecture

The model uses a custom architecture with:
- Convolutional layers for feature extraction
- Batch normalization for training stability
- Custom `AccuratePredictionLayer` that combines:
  - Color analysis
  - Texture analysis
  - Disease-specific pattern recognition

## Usage

To regenerate the mock models:

```bash
python mock_model.py
```

This will create both standard and quantized TFLite models in the current directory.

## Integration

The mock models can be used with the AgriScan web application by placing them in the appropriate directory. The application will automatically load the models for inference.

## Model Performance

- Standard model size: ~130 KB
- Quantized model size: ~100 KB
- Provides realistic predictions based on image characteristics
- Suitable for testing and demonstration purposes