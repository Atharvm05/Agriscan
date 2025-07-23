@echo off
echo Starting Enhanced Plant Disease Detection Model Training with virtual environment...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment and install dependencies
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if dependencies are installed
echo Checking dependencies...
pip show tensorflow >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

:: Run the dataset combination script if needed
if not exist "data\processed\combined\train" (
    echo Combining datasets...
    python model\combine_datasets.py
    if %errorlevel% neq 0 (
        echo Failed to combine datasets. Please check the error message above.
        pause
        exit /b 1
    )
)

:: Run the enhanced training script
echo Starting the enhanced model training...
python model\enhanced_train.py

echo Training completed. Press any key to exit.
pause