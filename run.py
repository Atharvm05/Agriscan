#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run script for AgriScan web application

This script starts the Flask web server for the AgriScan application.
It can be used for both development and production environments.
"""

import os
import argparse
from pathlib import Path

# Set the base directory
BASE_DIR = Path(__file__).resolve().parent

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the AgriScan web application')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--production', action='store_true', help='Run in production mode with gunicorn')
    args = parser.parse_args()
    
    # Change to the webapp directory
    webapp_dir = BASE_DIR / 'webapp'
    os.chdir(webapp_dir)
    
    if args.production:
        # Run with gunicorn for production
        workers = os.cpu_count() * 2 + 1  # Recommended number of workers
        cmd = f"gunicorn --workers={workers} --bind={args.host}:{args.port} app:app"
        print(f"Starting production server with command: {cmd}")
        os.system(cmd)
    else:
        # Run with Flask's built-in server for development
        # Try to import the real app, fall back to mock app if TensorFlow is not available
        import sys
        sys.path.append(str(webapp_dir))
        try:
            from app import app
            print("Using the real application with TensorFlow")
        except ImportError as e:
            print(f"Warning: {e}")
            print("Falling back to mock application without TensorFlow")
            from app_mock import app
            print("Using mock application for testing purposes only")
            print("Note: Model inference will return random results")
            print("Install TensorFlow to use the full application")
        print(f"Starting development server at http://{args.host}:{args.port}/")
        # When in debug mode, use the full path to this script for reloading
        if args.debug:
            # Save the current directory to return to it after the app runs
            original_dir = os.getcwd()
            # Change back to the base directory
            os.chdir(BASE_DIR)
            # Run the app with the full path to ensure proper reloading
            app.run(host=args.host, port=args.port, debug=args.debug)
            # Change back to the original directory
            os.chdir(original_dir)
        else:
            app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()