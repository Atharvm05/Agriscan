import os
import requests
import zipfile
import shutil
import tarfile
import gzip
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
import deeplake

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'model'

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Function to download and extract a file
def download_and_extract(url, output_path, extract_path=None):
    if extract_path is None:
        extract_path = output_path.parent
    
    # Download the file if it doesn't exist
    if not output_path.exists():
        print(f"Downloading {url} to {output_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"File {output_path} already exists.")
    
    # Extract the file based on its extension
    if output_path.suffix == '.zip':
        print(f"Extracting {output_path} to {extract_path}...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif output_path.suffix == '.tar':
        print(f"Extracting {output_path} to {extract_path}...")
        with tarfile.open(output_path, 'r') as tar_ref:
            tar_ref.extractall(extract_path)
    elif output_path.suffix == '.gz' and output_path.stem.endswith('.tar'):
        print(f"Extracting {output_path} to {extract_path}...")
        with tarfile.open(output_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)

# Function to download PlantVillage dataset
def download_plantvillage():
    # Check if dataset already exists
    if (RAW_DATA_DIR / 'PlantVillage').exists():
        print("PlantVillage dataset already downloaded.")
        return
    
    print("\nIMPORTANT: The PlantVillage dataset needs to be manually downloaded from Kaggle.")
    print("Please download the dataset from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print(f"Extract the downloaded zip file to: {RAW_DATA_DIR}")
    print("After downloading, the directory structure should be: data/raw/PlantVillage/")
    print("\nPress Enter to continue once the dataset is downloaded and extracted...")
    input()
    
    # Verify dataset exists
    if not (RAW_DATA_DIR / 'PlantVillage').exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_DIR / 'PlantVillage'}. Please download and extract the dataset.")

# Function to download DiaMOS Plant dataset
def download_diamos_plant():
    # Check if dataset already exists
    if (RAW_DATA_DIR / 'DiaMOS_Plant').exists():
        print("DiaMOS Plant dataset already downloaded.")
        return
    
    print("\nDownloading DiaMOS Plant dataset...")
    print("This dataset contains 3505 images of pear fruit and leaves affected by four diseases.")
    
    # The dataset is available on Zenodo
    url = "https://zenodo.org/record/5188491/files/DiaMOS_Plant.zip"
    output_path = RAW_DATA_DIR / "DiaMOS_Plant.zip"
    
    try:
        download_and_extract(url, output_path, RAW_DATA_DIR)
        print("DiaMOS Plant dataset downloaded and extracted successfully.")
    except Exception as e:
        print(f"Error downloading DiaMOS Plant dataset: {e}")
        print("\nPlease download the dataset manually from: https://zenodo.org/record/5188491")
        print(f"Extract the downloaded zip file to: {RAW_DATA_DIR / 'DiaMOS_Plant'}")
        print("\nPress Enter to continue...")
        input()

# Function to download Rice Leaf Disease dataset
def download_rice_leaf_disease():
    # Check if dataset already exists
    if (RAW_DATA_DIR / 'Rice_Leaf_Diseases').exists():
        print("Rice Leaf Disease dataset already downloaded.")
        return
    
    print("\nIMPORTANT: The Rice Leaf Disease dataset needs to be manually downloaded from Kaggle.")
    print("Please download the dataset from: https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases-detection-dataset")
    print(f"Extract the downloaded zip file to: {RAW_DATA_DIR / 'Rice_Leaf_Diseases'}")
    print("\nPress Enter to continue once the dataset is downloaded and extracted...")
    input()
    
    # Verify dataset exists
    if not (RAW_DATA_DIR / 'Rice_Leaf_Diseases').exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_DIR / 'Rice_Leaf_Diseases'}. Please download and extract the dataset.")

# Function to download Coffee Leaf Disease dataset
def download_coffee_leaf_disease():
    # Check if dataset already exists
    if (RAW_DATA_DIR / 'Coffee_Leaf_Diseases').exists():
        print("Coffee Leaf Disease dataset already downloaded.")
        return
    
    print("\nIMPORTANT: The Coffee Leaf Disease dataset needs to be manually downloaded.")
    print("Please download the JMuBEN dataset from: https://data.mendeley.com/datasets/yy2k5y8mxg/1")
    print(f"Extract the downloaded zip file to: {RAW_DATA_DIR / 'Coffee_Leaf_Diseases'}")
    print("\nPress Enter to continue once the dataset is downloaded and extracted...")
    input()
    
    # Verify dataset exists
    if not (RAW_DATA_DIR / 'Coffee_Leaf_Diseases').exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_DIR / 'Coffee_Leaf_Diseases'}. Please download and extract the dataset.")

# Function to download Maize Disease dataset
def download_maize_disease():
    # Check if dataset already exists
    if (RAW_DATA_DIR / 'Maize_Disease').exists():
        print("Maize Disease dataset already downloaded.")
        return
    
    print("\nIMPORTANT: The Maize Disease dataset needs to be manually downloaded.")
    print("Please download the dataset from: https://github.com/aldrin233/maize-disease-detection-dataset")
    print(f"Extract the downloaded zip file to: {RAW_DATA_DIR / 'Maize_Disease'}")
    print("\nPress Enter to continue once the dataset is downloaded and extracted...")
    input()
    
    # Verify dataset exists
    if not (RAW_DATA_DIR / 'Maize_Disease').exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_DIR / 'Maize_Disease'}. Please download and extract the dataset.")

# Function to load PlantVillage dataset using TensorFlow Datasets
def load_plantvillage_tfds():
    try:
        print("Loading PlantVillage dataset using TensorFlow Datasets...")
        (ds_train,), ds_info = tfds.load(
            name='plant_village',
            split=['train'],
            with_info=True,
            as_supervised=True
        )
        print(f"PlantVillage dataset loaded successfully with {ds_info.splits['train'].num_examples} examples.")
        return ds_train, ds_info
    except Exception as e:
        print(f"Error loading PlantVillage dataset using TensorFlow Datasets: {e}")
        print("Falling back to manual download method.")
        download_plantvillage()
        return None, None

# Function to load PlantVillage dataset using DeepLake
def load_plantvillage_deeplake():
    try:
        print("Loading PlantVillage dataset using DeepLake...")
        ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')
        print(f"PlantVillage dataset loaded successfully with {len(ds)} examples.")
        return ds
    except Exception as e:
        print(f"Error loading PlantVillage dataset using DeepLake: {e}")
        print("Falling back to manual download method.")
        download_plantvillage()
        return None

# Main function to download all datasets
def main():
    print("Starting dataset download process...")
    
    # Try to load PlantVillage using TensorFlow Datasets or DeepLake
    ds_tfds, ds_info = load_plantvillage_tfds()
    if ds_tfds is None:
        ds_deeplake = load_plantvillage_deeplake()
    
    # Download additional datasets
    download_diamos_plant()
    download_rice_leaf_disease()
    download_coffee_leaf_disease()
    download_maize_disease()
    
    print("\nAll datasets downloaded successfully!")
    print("You can now run the train.py script to train the model with the combined datasets.")

if __name__ == "__main__":
    main()