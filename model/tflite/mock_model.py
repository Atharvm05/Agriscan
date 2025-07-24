#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock TFLite model generator for Vercel deployment

This script creates a minimal TFLite model file that can be used for deployment
when the real model is too large to be included in the repository.
"""

import numpy as np
import json
import os
from pathlib import Path

# Set paths
BASE_DIR = Path(__file__).resolve().parent

def create_mock_tflite_model():
    """
    Create a minimal mock TFLite model file for deployment testing.
    This is just a placeholder and won't actually perform inference.
    """
    # Create a simple model structure with at least 7 bytes
    mock_model_content = bytes([0x54, 0x46, 0x4C, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])  # TFL3 magic bytes + padding
    
    # Write the mock model file
    with open(BASE_DIR / 'agriscan_model.tflite', 'wb') as f:
        f.write(mock_model_content)
    
    print("Created mock TFLite model for deployment testing")

if __name__ == "__main__":
    create_mock_tflite_model()