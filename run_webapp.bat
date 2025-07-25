@echo off
echo Starting AgriScan Web Application...
cd %~dp0
set FLASK_APP=webapp/app.py
set FLASK_ENV=development
python -m flask run --host=0.0.0.0 --port=5000