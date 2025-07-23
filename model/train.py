import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import requests
import zipfile
from pathlib import Path

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

# Configuration parameters
IMAGE_SIZE = 224  # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Function to download and extract the PlantVillage dataset
def download_dataset():
    # Check if dataset already exists
    if (RAW_DATA_DIR / 'PlantVillage').exists():
        print("Dataset already downloaded.")
        return
    
    print("Downloading PlantVillage dataset...")
    # Note: In a real implementation, you would need to provide the actual download URL
    # Since the Kaggle dataset requires authentication, users will need to manually download
    # the dataset and place it in the raw data directory
    print("\nIMPORTANT: The PlantVillage dataset needs to be manually downloaded from Kaggle.")
    print("Please download the dataset from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print(f"Extract the downloaded zip file to: {RAW_DATA_DIR}")
    print("After downloading, the directory structure should be: data/raw/PlantVillage/")
    print("\nPress Enter to continue once the dataset is downloaded and extracted...")
    input()
    
    # Verify dataset exists
    if not (RAW_DATA_DIR / 'PlantVillage').exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_DIR / 'PlantVillage'}. Please download and extract the dataset.")

# Function to prepare the dataset
def prepare_dataset():
    # Create train, validation, and test directories
    train_dir = PROCESSED_DATA_DIR / 'train'
    val_dir = PROCESSED_DATA_DIR / 'validation'
    test_dir = PROCESSED_DATA_DIR / 'test'
    
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
    
    # Get all class directories
    class_dirs = [d for d in (RAW_DATA_DIR / 'PlantVillage').iterdir() if d.is_dir()]
    
    # Create class directories in train, validation, and test directories
    for class_dir in class_dirs:
        class_name = class_dir.name
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all images in the class directory
        images = list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.jpg'))
        np.random.shuffle(images)
        
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
    
    print(f"Dataset prepared with {len(class_dirs)} classes.")
    return train_dir, val_dir, test_dir

# Function to create data generators with augmentation
def create_data_generators(train_dir, val_dir, test_dir):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

# Function to build the model
def build_model(num_classes):
    # Load the MobileNetV2 model without the top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# Function to train the model
def train_model(model, train_generator, validation_generator):
    # Define callbacks
    checkpoint = ModelCheckpoint(
        MODEL_DIR / 'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

# Function to fine-tune the model
def fine_tune_model(model, base_model, train_generator, validation_generator):
    # Unfreeze some layers of the base model
    for layer in base_model.layers[-30:]:  # Unfreeze the last 30 layers
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        MODEL_DIR / 'best_fine_tuned_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=10,  # Fewer epochs for fine-tuning
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

# Function to evaluate the model
def evaluate_model(model, test_generator):
    # Evaluate the model on the test set
    results = model.evaluate(test_generator)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    # Get predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    # Save class labels to a file
    with open(MODEL_DIR / 'class_labels.txt', 'w') as f:
        for label in class_labels:
            f.write(f"{label}\n")
    
    # Save model summary
    with open(MODEL_DIR / 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    return results, predicted_classes, true_classes, class_labels

# Function to plot training history
def plot_training_history(history, fine_tune_history=None):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if fine_tune_history:
        # Get the last accuracy values from the first training
        last_acc = history.history['accuracy'][-1]
        last_val_acc = history.history['val_accuracy'][-1]
        # Plot fine-tuning accuracy starting from the last values
        epochs_first = len(history.history['accuracy'])
        epochs_second = len(fine_tune_history.history['accuracy'])
        epochs_range = range(epochs_first, epochs_first + epochs_second)
        plt.plot(epochs_range, fine_tune_history.history['accuracy'], label='Fine-tuning Training Accuracy')
        plt.plot(epochs_range, fine_tune_history.history['val_accuracy'], label='Fine-tuning Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if fine_tune_history:
        # Get the last loss values from the first training
        last_loss = history.history['loss'][-1]
        last_val_loss = history.history['val_loss'][-1]
        # Plot fine-tuning loss starting from the last values
        plt.plot(epochs_range, fine_tune_history.history['loss'], label='Fine-tuning Training Loss')
        plt.plot(epochs_range, fine_tune_history.history['val_loss'], label='Fine-tuning Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'training_history.png')
    plt.close()

# Main function
def main():
    print("Starting AgriScan model training...")
    
    # Download and prepare dataset
    download_dataset()
    train_dir, val_dir, test_dir = prepare_dataset()
    
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators(train_dir, val_dir, test_dir)
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    
    # Build the model
    model, base_model = build_model(num_classes)
    print("Model built successfully.")
    
    # Train the model
    print("\nTraining the model...")
    history = train_model(model, train_generator, validation_generator)
    print("Initial training completed.")
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    fine_tune_history = fine_tune_model(model, base_model, train_generator, validation_generator)
    print("Fine-tuning completed.")
    
    # Evaluate the model
    print("\nEvaluating the model...")
    results, predicted_classes, true_classes, class_labels = evaluate_model(model, test_generator)
    
    # Plot training history
    plot_training_history(history, fine_tune_history)
    
    # Save the final model
    model.save(MODEL_DIR / 'agriscan_model.h5')
    print(f"Model saved to {MODEL_DIR / 'agriscan_model.h5'}")
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()