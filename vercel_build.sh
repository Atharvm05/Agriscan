#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p webapp/static/uploads
mkdir -p model/tflite

# Set up environment for Vercel
export FLASK_APP=webapp/app.py
export FLASK_ENV=production

# Note: The actual model files need to be uploaded separately or generated during build
# This script assumes the model files are already available or will be generated
# in a separate step

echo "Build completed successfully"