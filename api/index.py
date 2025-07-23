from flask import Flask, redirect
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import from webapp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Vercel-specific app implementation
from api.app_vercel import app

# This is for Vercel serverless deployment
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))