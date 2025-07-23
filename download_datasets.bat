@echo off
echo Starting Plant Disease Datasets Download with virtual environment...

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

:: Run the dataset download script
echo Starting the datasets download...
python model\download_datasets.py

echo.
echo Download process completed.
echo You can now run train_enhanced_model.bat to train the model with the combined datasets.
echo Press any key to exit.
pause