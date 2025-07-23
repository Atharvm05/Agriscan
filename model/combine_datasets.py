import os
import numpy as np
import tensorflow as tf
import shutil
from pathlib import Path
import random
from PIL import Image
import pandas as pd

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
COMBINED_DATA_DIR = PROCESSED_DATA_DIR / 'combined'

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define dataset paths
PLANTVILLAGE_DIR = RAW_DATA_DIR / 'PlantVillage'
DIAMOS_PLANT_DIR = RAW_DATA_DIR / 'DiaMOS_Plant'
RICE_LEAF_DIR = RAW_DATA_DIR / 'Rice_Leaf_Diseases'
COFFEE_LEAF_DIR = RAW_DATA_DIR / 'Coffee_Leaf_Diseases'
MAIZE_DISEASE_DIR = RAW_DATA_DIR / 'Maize_Disease'

# Function to create train, validation, and test directories
def create_split_directories():
    # Create train, validation, and test directories
    train_dir = COMBINED_DATA_DIR / 'train'
    val_dir = COMBINED_DATA_DIR / 'validation'
    test_dir = COMBINED_DATA_DIR / 'test'
    
    # Remove existing directories if they exist
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    return train_dir, val_dir, test_dir

# Function to process PlantVillage dataset
def process_plantvillage(train_dir, val_dir, test_dir):
    if not PLANTVILLAGE_DIR.exists():
        print(f"PlantVillage dataset not found at {PLANTVILLAGE_DIR}. Skipping...")
        return 0
    
    print("Processing PlantVillage dataset...")
    
    # Get all class directories
    class_dirs = [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir()]
    
    total_images = 0
    
    # Process each class directory
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Create class directories in train, validation, and test directories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all images in the class directory
        images = list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.jpg'))
        random.shuffle(images)
        
        # Split images into train (70%), validation (15%), and test (15%)
        n_train = int(0.7 * len(images))
        n_val = int(0.15 * len(images))
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / class_name / img.name)
        for img in test_images:
            shutil.copy(img, test_dir / class_name / img.name)
        
        total_images += len(images)
    
    print(f"Processed {len(class_dirs)} classes and {total_images} images from PlantVillage dataset.")
    return total_images

# Function to process DiaMOS Plant dataset
def process_diamos_plant(train_dir, val_dir, test_dir):
    if not DIAMOS_PLANT_DIR.exists():
        print(f"DiaMOS Plant dataset not found at {DIAMOS_PLANT_DIR}. Skipping...")
        return 0
    
    print("Processing DiaMOS Plant dataset...")
    
    # Get all class directories
    class_dirs = [d for d in DIAMOS_PLANT_DIR.iterdir() if d.is_dir()]
    
    total_images = 0
    
    # Process each class directory
    for class_dir in class_dirs:
        class_name = f"Pear_{class_dir.name}"  # Prefix with 'Pear_' to avoid conflicts
        
        # Create class directories in train, validation, and test directories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all images in the class directory
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.PNG'))
        random.shuffle(images)
        
        # Split images into train (70%), validation (15%), and test (15%)
        n_train = int(0.7 * len(images))
        n_val = int(0.15 * len(images))
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / class_name / img.name)
        for img in test_images:
            shutil.copy(img, test_dir / class_name / img.name)
        
        total_images += len(images)
    
    print(f"Processed {len(class_dirs)} classes and {total_images} images from DiaMOS Plant dataset.")
    return total_images

# Function to process Rice Leaf Disease dataset
def process_rice_leaf_disease(train_dir, val_dir, test_dir):
    if not RICE_LEAF_DIR.exists():
        print(f"Rice Leaf Disease dataset not found at {RICE_LEAF_DIR}. Skipping...")
        return 0
    
    print("Processing Rice Leaf Disease dataset...")
    
    # Get all class directories
    class_dirs = [d for d in RICE_LEAF_DIR.iterdir() if d.is_dir()]
    
    total_images = 0
    
    # Process each class directory
    for class_dir in class_dirs:
        class_name = f"Rice_{class_dir.name}"  # Prefix with 'Rice_' to avoid conflicts
        
        # Create class directories in train, validation, and test directories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all images in the class directory
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.PNG'))
        random.shuffle(images)
        
        # Split images into train (70%), validation (15%), and test (15%)
        n_train = int(0.7 * len(images))
        n_val = int(0.15 * len(images))
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / class_name / img.name)
        for img in test_images:
            shutil.copy(img, test_dir / class_name / img.name)
        
        total_images += len(images)
    
    print(f"Processed {len(class_dirs)} classes and {total_images} images from Rice Leaf Disease dataset.")
    return total_images

# Function to process Coffee Leaf Disease dataset
def process_coffee_leaf_disease(train_dir, val_dir, test_dir):
    if not COFFEE_LEAF_DIR.exists():
        print(f"Coffee Leaf Disease dataset not found at {COFFEE_LEAF_DIR}. Skipping...")
        return 0
    
    print("Processing Coffee Leaf Disease dataset...")
    
    # Get all class directories
    class_dirs = [d for d in COFFEE_LEAF_DIR.iterdir() if d.is_dir()]
    
    total_images = 0
    
    # Process each class directory
    for class_dir in class_dirs:
        class_name = f"Coffee_{class_dir.name}"  # Prefix with 'Coffee_' to avoid conflicts
        
        # Create class directories in train, validation, and test directories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all images in the class directory
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.PNG'))
        random.shuffle(images)
        
        # Split images into train (70%), validation (15%), and test (15%)
        n_train = int(0.7 * len(images))
        n_val = int(0.15 * len(images))
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / class_name / img.name)
        for img in test_images:
            shutil.copy(img, test_dir / class_name / img.name)
        
        total_images += len(images)
    
    print(f"Processed {len(class_dirs)} classes and {total_images} images from Coffee Leaf Disease dataset.")
    return total_images

# Function to process Maize Disease dataset
def process_maize_disease(train_dir, val_dir, test_dir):
    if not MAIZE_DISEASE_DIR.exists():
        print(f"Maize Disease dataset not found at {MAIZE_DISEASE_DIR}. Skipping...")
        return 0
    
    print("Processing Maize Disease dataset...")
    
    # Get all class directories
    class_dirs = [d for d in MAIZE_DISEASE_DIR.iterdir() if d.is_dir()]
    
    total_images = 0
    
    # Process each class directory
    for class_dir in class_dirs:
        class_name = f"Maize_{class_dir.name}"  # Prefix with 'Maize_' to avoid conflicts
        
        # Create class directories in train, validation, and test directories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all images in the class directory
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.PNG'))
        random.shuffle(images)
        
        # Split images into train (70%), validation (15%), and test (15%)
        n_train = int(0.7 * len(images))
        n_val = int(0.15 * len(images))
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / class_name / img.name)
        for img in test_images:
            shutil.copy(img, test_dir / class_name / img.name)
        
        total_images += len(images)
    
    print(f"Processed {len(class_dirs)} classes and {total_images} images from Maize Disease dataset.")
    return total_images

# Function to create a class mapping file
def create_class_mapping(train_dir):
    # Get all class directories
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    
    # Create a mapping from class index to class name
    class_mapping = {}
    for i, class_dir in enumerate(sorted(class_dirs)):
        class_mapping[i] = class_dir.name
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(list(class_mapping.items()), columns=['index', 'class_name'])
    df.to_csv(COMBINED_DATA_DIR / 'class_mapping.csv', index=False)
    
    # Also save as a text file for compatibility
    with open(COMBINED_DATA_DIR / 'class_labels.txt', 'w') as f:
        for i in range(len(class_mapping)):
            f.write(f"{class_mapping[i]}\n")
    
    print(f"Created class mapping with {len(class_mapping)} classes.")
    return class_mapping

# Main function to combine all datasets
def main():
    print("Starting dataset combination process...")
    
    # Create split directories
    train_dir, val_dir, test_dir = create_split_directories()
    
    # Process each dataset
    total_images = 0
    total_images += process_plantvillage(train_dir, val_dir, test_dir)
    total_images += process_diamos_plant(train_dir, val_dir, test_dir)
    total_images += process_rice_leaf_disease(train_dir, val_dir, test_dir)
    total_images += process_coffee_leaf_disease(train_dir, val_dir, test_dir)
    total_images += process_maize_disease(train_dir, val_dir, test_dir)
    
    # Create class mapping
    class_mapping = create_class_mapping(train_dir)
    
    print(f"\nCombined dataset created successfully with {len(class_mapping)} classes and {total_images} images.")
    print(f"Dataset is located at {COMBINED_DATA_DIR}")
    print("You can now run the enhanced_train.py script to train the model with the combined dataset.")

if __name__ == "__main__":
    main()