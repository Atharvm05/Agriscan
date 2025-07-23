import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import random
import shutil
import time

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
COMBINED_DATA_DIR = PROCESSED_DATA_DIR / 'combined'
MODEL_DIR = BASE_DIR / 'model'

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configuration parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001

# Check if combined dataset exists
def check_combined_dataset():
    train_dir = COMBINED_DATA_DIR / 'train'
    val_dir = COMBINED_DATA_DIR / 'validation'
    test_dir = COMBINED_DATA_DIR / 'test'
    
    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        print("Combined dataset not found. Please run combine_datasets.py first.")
        print("Command: python model/combine_datasets.py")
        return False
    
    # Check if there are class directories in the train directory
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if len(class_dirs) == 0:
        print("No class directories found in the combined dataset. Please run combine_datasets.py first.")
        print("Command: python model/combine_datasets.py")
        return False
    
    print(f"Found combined dataset with {len(class_dirs)} classes.")
    return True

# Create data generators with augmentation
def create_data_generators():
    train_dir = COMBINED_DATA_DIR / 'train'
    val_dir = COMBINED_DATA_DIR / 'validation'
    test_dir = COMBINED_DATA_DIR / 'test'
    
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

# Build the model
def build_model(num_classes):
    # Load the MobileNetV2 model with pre-trained weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
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

# Train the model
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
        min_lr=0.00001,
        verbose=1
    )
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

# Fine-tune the model
def fine_tune_model(model, base_model, train_generator, validation_generator):
    # Unfreeze the last 30 layers of the base model
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
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
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return fine_tune_history

# Evaluate the model
def evaluate_model(model, test_generator):
    print("\nEvaluating the model...")
    loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Get predictions
    predictions = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = test_generator.classes[:len(predicted_classes)]
    
    # Get class labels
    class_labels = list(test_generator.class_indices.keys())
    
    # Save class labels to a file
    with open(MODEL_DIR / 'class_labels.txt', 'w') as f:
        for label in class_labels:
            f.write(f"{label}\n")
    
    # Save model summary to a file
    with open(MODEL_DIR / 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    return accuracy, class_labels

# Plot training history
def plot_training_history(history, fine_tune_history=None):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    
    if fine_tune_history is not None:
        # Plot fine-tuning accuracy values
        plt.plot(np.arange(len(history.history['accuracy']), len(history.history['accuracy']) + len(fine_tune_history.history['accuracy'])), fine_tune_history.history['accuracy'])
        plt.plot(np.arange(len(history.history['val_accuracy']), len(history.history['val_accuracy']) + len(fine_tune_history.history['val_accuracy'])), fine_tune_history.history['val_accuracy'])
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Fine-tune Train', 'Fine-tune Validation'], loc='lower right')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    if fine_tune_history is not None:
        # Plot fine-tuning loss values
        plt.plot(np.arange(len(history.history['loss']), len(history.history['loss']) + len(fine_tune_history.history['loss'])), fine_tune_history.history['loss'])
        plt.plot(np.arange(len(history.history['val_loss']), len(history.history['val_loss']) + len(fine_tune_history.history['val_loss'])), fine_tune_history.history['val_loss'])
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Fine-tune Train', 'Fine-tune Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'training_history.png')
    plt.close()

# Main function
def main():
    print("Enhanced Plant Disease Detection Model Training")
    print("==============================================\n")
    
    # Check if combined dataset exists
    if not check_combined_dataset():
        return
    
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators()
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    
    # Build the model
    model, base_model = build_model(num_classes)
    print("Model built successfully.")
    
    # Train the model
    start_time = time.time()
    history = train_model(model, train_generator, validation_generator)
    
    # Fine-tune the model
    fine_tune_history = fine_tune_model(model, base_model, train_generator, validation_generator)
    
    # Evaluate the model
    accuracy, class_labels = evaluate_model(model, test_generator)
    
    # Plot training history
    plot_training_history(history, fine_tune_history)
    
    # Save the final model
    model.save(MODEL_DIR / 'agriscan_model.h5')
    print(f"\nModel saved to {MODEL_DIR / 'agriscan_model.h5'}")
    
    # Print training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    print(f"\nTraining completed successfully with {accuracy:.2%} accuracy on test set.")
    print(f"The model can classify {num_classes} different plant diseases:")
    for i, label in enumerate(class_labels):
        print(f"  {i+1}. {label}")

if __name__ == "__main__":
    main()