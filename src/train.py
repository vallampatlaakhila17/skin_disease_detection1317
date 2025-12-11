# src/train.py - Optimized for High Accuracy

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.utils import class_weight

# --- CRITICAL FIX: Standard Imports for relative modules ---
from data_loader import load_and_preprocess_data, NUM_CLASSES 
from model_architecture import create_attention_cnn_model 

# --- Training Configuration ---
BATCH_SIZE = 32
EPOCHS = 100 
MODEL_OUTPUT_PATH = 'C:/Users/rakhi/Desktop/major/pottulu/models/best_high_accuracy_model.h5' 

def compile_and_train():
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data()
    
    if X_train is None:
        print("\n❌ Training aborted due to data loading error.")
        return

    # --- CLASS WEIGHTING ---
    y_integers = np.argmax(Y_train, axis=1)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weights = dict(enumerate(weights))

    # Build Model
    model = create_attention_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), num_classes=NUM_CLASSES)
    
    # --- DATA AUGMENTATION ---
    datagen = ImageDataGenerator(
        rotation_range=20, zoom_range=0.15, width_shift_range=0.1, 
        height_shift_range=0.1, shear_range=0.1, horizontal_flip=True, 
        vertical_flip=True, fill_mode="nearest"
    )
    datagen.fit(X_train)

    # Define Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(
        filepath=MODEL_OUTPUT_PATH, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1
    )
    lr_reducer = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )
    callbacks_list = [model_checkpoint, early_stopping, lr_reducer]

    # Compile Model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train Model
    print("\n--- Starting Optimized Model Training ---")
    model.fit(
        datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, Y_test),
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("\nTraining Complete.")

if __name__ == '__main__':
    os.makedirs('C:/Users/rakhi/Desktop/major/pottulu/models', exist_ok=True)
    compile_and_train()