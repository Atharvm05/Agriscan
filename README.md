# AgriScan - Crop Disease Detection with Offline Mode

AgriScan is a machine learning web application that allows farmers to take photos of crop leaves and detect diseases, even in offline mode using TensorFlow Lite and TensorFlow.js. The application is designed as a Progressive Web App (PWA) that can be installed on mobile devices and used without an internet connection.

## Features

- **Disease Detection**: Upload or capture images of plant leaves to identify diseases
- **Offline Mode**: Works without internet connection using TensorFlow Lite and TensorFlow.js
- **Progressive Web App (PWA)**: Can be installed on mobile devices and used like a native app
- **Camera Integration**: Capture photos directly within the app using device camera
- **Treatment Recommendations**: Provides actionable advice for detected diseases
- **Mobile-Friendly**: Responsive design works on smartphones and tablets
- **Background Sync**: Queues uploads when offline and processes them when back online

## Technology Stack

- **Model**: TensorFlow CNN / MobileNetV2 for image classification
- **Dataset**: PlantVillage dataset with various crop disease images
- **Backend**: Flask web server
- **Frontend**: HTML, CSS, JavaScript
- **Offline Capability**: 
  - TensorFlow Lite for model compression
  - TensorFlow.js for in-browser inference
  - Service Workers for offline asset caching
  - IndexedDB for client-side storage
- **PWA Features**: 
  - Installable on mobile devices
  - Works offline
  - Background synchronization
  - Push notifications (optional)

## Project Structure

```
.
├── data/                      # Dataset files
│   ├── raw/                   # Original datasets
│   │   ├── PlantVillage/      # PlantVillage dataset
│   │   ├── DiaMOS_Plant/      # DiaMOS Plant dataset
│   │   ├── Rice_Leaf_Diseases/ # Rice Leaf Disease dataset
│   │   ├── Coffee_Leaf_Diseases/ # Coffee Leaf Disease dataset
│   │   └── Maize_Disease/     # Maize Disease dataset
│   └── processed/             # Processed and augmented images
│       ├── train/             # Training data split (standard model)
│       ├── validation/        # Validation data split (standard model)
│       ├── test/              # Test data split (standard model)
│       └── combined/          # Combined datasets (enhanced model)
│           ├── train/         # Training data from all datasets
│           ├── validation/    # Validation data from all datasets
│           └── test/          # Test data from all datasets
├── model/                     # Model training and evaluation
│   ├── train.py               # Script for training the standard CNN model
│   ├── download_datasets.py   # Script for downloading multiple datasets
│   ├── combine_datasets.py    # Script for combining multiple datasets
│   ├── enhanced_train.py      # Script for training with multiple datasets
│   ├── evaluate.py            # Model evaluation script
│   └── convert.py             # Convert to TensorFlow Lite and TensorFlow.js
├── webapp/                    # Flask web application
│   ├── app.py                 # Main Flask application
│   ├── static/                # Static files
│   │   ├── css/               # Stylesheets
│   │   │   └── style.css      # Main stylesheet
│   │   ├── js/                # JavaScript files
│   │   │   ├── app.js         # Main application logic
│   │   │   ├── database.js    # IndexedDB for offline storage
│   │   │   ├── offline-inference.js # TensorFlow.js inference
│   │   │   └── service-worker.js # Service worker for PWA
│   │   ├── images/            # Images and icons
│   │   ├── data/              # Offline data (disease info)
│   │   └── manifest.json      # PWA manifest
│   └── templates/             # HTML templates
│       ├── base.html          # Base template
│       ├── index.html         # Home page
│       ├── about.html         # About page
│       └── offline.html       # Offline mode page
├── requirements.txt           # Python dependencies
├── run.py                    # Main runner script
├── run.bat                   # Windows batch file for running the app
├── run.sh                    # Shell script for running the app on Unix
├── run_with_venv.bat         # Windows batch file with virtual environment
├── download_datasets.bat     # Windows batch file for downloading datasets
├── train_enhanced_model.bat  # Windows batch file for enhanced training
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the repository**

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Download the datasets**
   - Option 1: Use the automated script (recommended)
     ```
     download_datasets.bat    # Windows
     ```
     or
     ```
     python model/download_datasets.py    # Manual execution
     ```
   - Option 2: Manually download datasets
     - PlantVillage dataset
     - DiaMOS Plant dataset
     - Rice Leaf Disease dataset
     - Coffee Leaf Disease dataset
     - Maize Disease dataset
     - Extract each to the appropriate directory under `data/raw`

4. **Train the model**
   - Option 1: Standard training with PlantVillage dataset only
     ```
     python model/train.py
     ```
   - Option 2: Enhanced training with multiple datasets (recommended)
     ```
     train_enhanced_model.bat    # Windows
     ```
     or
     ```
     python model/enhanced_train.py    # Manual execution
     ```

5. **Convert the model**
   ```
   python model/convert.py
   ```
   This will create:
   - TensorFlow Lite model (`.tflite`) for mobile deployment
   - TensorFlow.js model files for browser-based inference

6. **Run the web application**
   ```
   run_with_venv.bat    # Windows with virtual environment
   ```
   or
   ```
   python run.py --debug    # Manual execution
   ```

7. **Access the application**
   - Open a web browser and go to `http://localhost:5000`
   
8. **Install as PWA (optional)**
   - In Chrome/Edge: Click the install icon in the address bar
   - In Safari on iOS: Tap the share button and select "Add to Home Screen"
   
9. **Test offline functionality**
   - After initial load, disconnect from the internet
   - The application should continue to work for image analysis
   - Uploads will be queued and processed when back online

## Model Training and Conversion

### Standard Model
The standard model is trained on the PlantVillage dataset, which contains images of healthy and diseased plant leaves across various crops. We use a Convolutional Neural Network (CNN) architecture based on MobileNetV2 for efficient deployment on mobile devices.

### Enhanced Model (Recommended)
The enhanced model is trained on multiple datasets to improve accuracy and robustness across different plant species and disease conditions:

- **PlantVillage Dataset**: 54,000+ images across 38 classes of crop diseases
- **DiaMOS Plant Dataset**: 3,500+ images of pear fruit and leaves with four diseases
- **Rice Leaf Disease Dataset**: Images of common rice leaf diseases
- **Coffee Leaf Disease Dataset**: Images of coffee leaf diseases including rust, miner, and red spider mites
- **Maize Disease Dataset**: 18,000+ field images of maize leaves with Northern Leaf Blight

The enhanced training process includes:
- Automatic dataset combination with proper class naming to avoid conflicts
- Consistent train/validation/test splitting (70%/15%/15%)
- Data augmentation for improved model generalization
- Transfer learning with MobileNetV2 architecture
- Fine-tuning of model parameters for optimal performance

### Model Conversion

The trained model is converted to multiple formats for different deployment scenarios:

1. **TensorFlow Lite (.tflite)**
   - Optimized for mobile and edge devices
   - Supports quantization for smaller file size
   - Used for native mobile applications

2. **TensorFlow.js (model.json + *.bin)**
   - Optimized for browser-based inference
   - Enables client-side prediction without server calls
   - Powers the offline functionality in the PWA

## Offline Functionality

The application works offline through several technologies:

1. **Service Worker**: Caches static assets (HTML, CSS, JS, images)
2. **IndexedDB**: Stores:
   - Pending image uploads
   - Analysis results
   - TensorFlow.js model and weights
   - Disease information database
3. **Background Sync**: Queues uploads when offline and processes them when online

## License

This project is licensed under the MIT License - see the LICENSE file for details.