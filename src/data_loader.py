# src/data_loader.py

import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- Configuration (Based on your project structure) ---
DATA_DIR = 'C:/Users/rakhi/Desktop/major/pottulu/data/images/' 
METADATA_PATH = 'C:/Users/rakhi/Desktop/major/pottulu/data/HAM10000_metadata.csv'

IMG_SIZE = 128
NUM_CLASSES = 7
# --------------------------------------------------------

def load_and_preprocess_data():
    """Loads the HAM10000 dataset, processes images, and splits data."""
    
    # --- 1. Load Metadata ---
    try:
        metadata = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Metadata file not found at: {METADATA_PATH}")
        return None, None, None, None
    
    metadata['cell_type_idx'] = metadata['dx'].astype('category').cat.codes
    
    image_list = []
    label_list = []
    
    # --- 2. Load and Preprocess Images ---
    for index, row in metadata.iterrows():
        img_name = row['image_id'] + '.jpg'
        img_path = os.path.join(DATA_DIR, img_name)
        
        try:
            img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
            img_array = np.asarray(img) / 255.0 # Normalization [0, 1]
            image_list.append(img_array)
            label_list.append(row['cell_type_idx'])
        except Exception:
            # Skip if image file is missing or corrupted
            continue
    
    X = np.array(image_list)
    Y_int = np.array(label_list)
    
    if len(X) == 0:
        print("\n❌ FATAL ERROR: No images were successfully loaded. Check DATA_DIR.")
        return None, None, None, None

    Y = to_categorical(Y_int, num_classes=NUM_CLASSES)
    
    # Stratified Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y_int
    )
    return X_train, X_test, Y_train, Y_test