# AgriScan Installation Guide

This guide will help you set up and run the AgriScan web application on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Clone or Download the Repository

If you have Git installed:

```bash
git clone https://github.com/yourusername/agriscan.git
cd agriscan
```

Or download and extract the ZIP file from the repository and navigate to the extracted folder.

### 2. Set Up a Virtual Environment (Recommended)

Creating a virtual environment helps isolate the project dependencies from your system Python installation.

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

This will install TensorFlow, Flask, and all other dependencies needed for the application.

### 4. Convert the Model (Optional)

If you have your own trained model or want to convert the provided model to TensorFlow Lite and TensorFlow.js formats:

```bash
python model/convert.py
```

This will create optimized models for web and mobile deployment in the appropriate directories.

### 5. Run the Application

#### Using the Provided Scripts

**Windows:**
Double-click on `run.bat` or run it from the command line:
```cmd
run.bat
```

**macOS/Linux:**
Make the script executable and run it:
```bash
chmod +x run.sh
./run.sh
```

#### Manual Execution

Alternatively, you can run the application directly:

```bash
python run.py --debug  # For development with debug mode
```

Or for production:

```bash
python run.py --production  # Uses gunicorn for production deployment
```

### 6. Access the Application

Open your web browser and navigate to:

```
http://localhost:5000
```

## Troubleshooting

### TensorFlow Installation Issues

If you encounter issues installing TensorFlow:

1. Make sure you have a compatible Python version (3.8-3.10 recommended for TensorFlow 2.x)
2. On Windows, you might need Microsoft Visual C++ Redistributable
3. For GPU support, ensure you have compatible CUDA and cuDNN versions

### Flask Application Not Starting

1. Check if the port 5000 is already in use by another application
2. Try specifying a different port: `python run.py --port 8000`

### Model Conversion Errors

1. Ensure TensorFlow and TensorFlow.js are properly installed
2. Check if the model file exists in the expected location

## Running as a Progressive Web App

To fully experience the PWA features:

1. Access the application through HTTPS (required for service workers)
2. For local testing, you can use tools like ngrok to create a temporary HTTPS URL
3. In Chrome/Edge, look for the install icon in the address bar to install as a PWA
4. On mobile devices, use the "Add to Home Screen" option in the browser menu

## Offline Functionality

After the first visit to the application:

1. The service worker will cache necessary resources
2. The TensorFlow.js model will be stored in the browser's IndexedDB
3. You can then use the application without an internet connection
4. Images captured while offline will be queued and processed when back online