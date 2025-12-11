# app/app.py - FINAL DEPLOYMENT SCRIPT WITH ALL FEATURES (Camera, Base64, Charts)

import os
import sys
import numpy as np
import tensorflow as tf
import cv2 
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from PIL import Image
import base64     
import io         

# --- CRITICAL FIX 1: Ensure 'src' directory is on the Python Path ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src') 
if src_dir not in sys.path:
    sys.path.append(src_dir)
# ------------------------------------------------------------------

# Import custom layer from src/
from model_architecture import channel_attention 

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads/'
IMG_SIZE = 128
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] 
# Note: The name of the final convolutional layer might differ based on model training. 
# 'conv2d_1' is used here based on previous conversation.
LAST_CONV_LAYER_NAME = 'conv2d_1' 
# ---------------------

# --- Model Path & Setup ---
PRIMARY_MODEL_PATH = os.path.join(project_root, 'models', 'best_high_accuracy_model.h5')
MODEL_PATH = PRIMARY_MODEL_PATH 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(os.path.join(current_dir, UPLOAD_FOLDER), exist_ok=True) 

# === DISEASE INFORMATION DATA (Expanded) ===
DISEASE_INFO = {
    'mel': {'name': 'Melanoma', 'desc': 'The most dangerous form of skin cancer. Characterized by ABCDE rules (Asymmetry, irregular Borders, varied Color, large Diameter, Evolving size/shape). Early detection is vital.', 'prev': 'Strict daily sun protection (SPF 30+). Monthly self-examination using the ABCDE criteria is essential.', 'treat': 'Surgical Excision, Mohs Surgery. For advanced stages: Immunotherapy, Targeted Therapy, or Chemotherapy.'},
    'nv': {'name': 'Melanocytic Nevus (Common Mole)', 'desc': 'A benign (non-cancerous) tumor of melanocytes. Typically small, round, uniform in color (brown or tan), and has well-defined borders.', 'prev': 'General sun protection to prevent new moles. Monitor for any suspicious changes (ABCDE).', 'treat': 'None required unless for cosmetic reasons or malignancy suspicion. Removed via surgical excision.'},
    'bcc': {'name': 'Basal Cell Carcinoma', 'desc': 'The most common form of skin cancer. Grows slowly and rarely spreads. Often appears as a waxy, pearly bump, or a persistent sore that bleeds easily.', 'prev': 'Consistent, year-round sun protection. Regular total body skin checks by a dermatologist.', 'treat': 'Surgical Excision, Mohs Micrographic Surgery, Curettage and Electrodesiccation. Topical creams like Imiquimod or 5-Fluorouracil for superficial BCC.'},
    'bkl': {'name': 'Benign Keratosis-like Lesions', 'desc': 'Common, harmless growths (e.g., Seborrheic Keratosis). Look like waxy, "stuck-on" warts, usually brown, black, or tan. Related to aging and genetics.', 'prev': 'No specific prevention, as they are related to aging.', 'treat': 'None required. Can be removed via Cryotherapy (freezing) or Curettage for cosmetic reasons or irritation.'},
    'akiec': {'name': 'Actinic Keratoses', 'desc': 'Pre-cancerous skin lesions in sun-damaged areas. Rough, scaly, crusty patches, red or tan. High risk of progressing into Squamous Cell Carcinoma (SCC).', 'prev': 'Mandatory, consistent, year-round use of broad-spectrum sunscreen. Protective hats and clothing.', 'treat': 'Cryotherapy (freezing with liquid nitrogen) is common. Topical Chemotherapy (5-FU) or Photodynamic Therapy (PDT) for widespread lesions.'},
    'df': {'name': 'Dermatofibroma', 'desc': 'A common, benign, firm, raised skin growth that often feels like a hard lump under the skin. Frequently occurs on the lower legs, often resulting from minor trauma.', 'prev': 'None specific, often due to minor trauma. Protecting skin from bites and scratches may help.', 'treat': 'None required. If removal is necessary due to pain or uncertain diagnosis, a Surgical Excision may be performed.'},
    'vasc': {'name': 'Vascular Lesions', 'desc': 'Benign lesions related to blood vessels (e.g., Angiomas). Typically bright red, purple, or blue bumps that blanch when pressed. Generally harmless and stable.', 'prev': 'None specific, as they are generally genetic or related to aging.', 'treat': 'None required. Can be removed for cosmetic reasons using Laser Therapy (e.g., pulsed dye laser) or electrosurgery.'}
}
# =========================================================

# === MODEL LOADING ===
try:
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
    model = load_model(
        MODEL_PATH, 
        custom_objects={'channel_attention': channel_attention}
    )
    print(f"✨ Model Loaded Successfully from: {MODEL_PATH} ✨")
except Exception as e:
    print(f"❌ Deployment failure: Could not load model. Reason: {e}")
    model = None

# === HELPER FUNCTIONS ===

def get_img_array(img_path, size):
    img = Image.open(img_path).resize(size)
    array = np.asarray(img)
    array = array.astype('float32') / 255.0
    return np.expand_dims(array, axis=0)

# --- GRAD-CAM HELPER (UNCHANGED) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = tf.cast(last_conv_layer_output, tf.float32) @ tf.cast(pooled_grads[..., tf.newaxis], tf.float32)
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    
    return heatmap.numpy()

# --- GRAD-CAM BASE64 GENERATOR (UNCHANGED) ---
def generate_gradcam_base64(img_path, heatmap, alpha=0.5):
    """Generates Grad-CAM, saves to a buffer, and returns a Base64 string."""
    
    img = cv2.imread(img_path)
    if img is None: return None
        
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img.astype(float), 1 - alpha, heatmap.astype(float), alpha, 0)
    
    is_success, buffer = cv2.imencode(".jpg", superimposed_img)
    if not is_success:
        return None
        
    base64_encoded = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{base64_encoded}" 

# --- CHANNEL WEIGHT EXTRACTOR (UNCHANGED) ---
def get_channel_attention_weights(img_array, model):
    """Extracts the channel-wise attention weights from the custom layer."""
    try:
        # Target the output layer of the attention block, which scales the features
        attention_output_layer = model.get_layer('multiply').output 
        
        attention_model = tf.keras.models.Model(
            [model.inputs], [attention_output_layer]
        )
        
        attention_feature_map = attention_model.predict(img_array)[0]
        
        # Calculate the weight/importance for each channel by averaging activation
        channel_weights = np.mean(np.abs(attention_feature_map), axis=(0, 1))
        
        # Normalize the weights
        max_weight = np.max(channel_weights)
        if max_weight != 0:
            normalized_weights = channel_weights / max_weight
        else:
            normalized_weights = channel_weights
            
        return normalized_weights.tolist()
        
    except ValueError as e:
        print(f"Warning: Could not extract channel weights (layer issue): {e}")
        return []

# --- RGB LAYER BASE64 GENERATOR (NEW) ---
def generate_rgb_base64(img_path):
    """Splits image into R, G, B channels and returns a list of Base64 URIs."""
    
    img = cv2.imread(img_path)
    if img is None: return [None, None, None]
        
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Split the BGR image (OpenCV default) into B, G, R channels
    B, G, R = cv2.split(img)
    
    rgb_base64_list = []
    
    for channel_data in [R, G, B]: # Iterate over R, G, B channel data
        # Create a blank image to display the grayscale channel intensity
        color_img = np.zeros_like(img)
        
        # Merge the single channel data back into all three channels to display as grayscale
        color_img[:,:,0] = channel_data 
        color_img[:,:,1] = channel_data 
        color_img[:,:,2] = channel_data 

        # Convert the processed image to Base64
        is_success, buffer = cv2.imencode(".jpg", color_img)
        if is_success:
            base64_encoded = base64.b64encode(buffer).decode()
            rgb_base64_list.append(f"data:image/jpeg;base64,{base64_encoded}")
        else:
            rgb_base64_list.append(None)
            
    # Returns [R_base64, G_base64, B_base64]
    return rgb_base64_list


# === FLASK ROUTES ===

def get_disease_info(key):
    return DISEASE_INFO.get(key, {'name': 'Unknown', 'desc': 'No information available.', 'prev': 'N/A', 'treat': 'N/A'})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and model:
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file temporarily
        temp_filename = "temp_" + file.filename
        filepath = os.path.join(current_dir, app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(filepath)
        
        # 1. PREDICTION
        processed_img = get_img_array(filepath, size=(IMG_SIZE, IMG_SIZE))
        preds = model.predict(processed_img)[0]
        top_index = np.argmax(preds)
        prediction_key = CLASS_NAMES[top_index]
        
        # 2. GRAD-CAM GENERATION
        heatmap = make_gradcam_heatmap(processed_img, model, LAST_CONV_LAYER_NAME, top_index)
        gradcam_base64_data = generate_gradcam_base64(filepath, heatmap)
        
        # 3. CHANNEL WEIGHT EXTRACTION
        channel_weights = get_channel_attention_weights(processed_img, model) 
        
        # 4. RGB LAYER VISUALIZATION
        rgb_layer_data = generate_rgb_base64(filepath) 
        
        # 5. CLEANUP - Remove the initial temporary uploaded file
        try:
             os.remove(filepath) 
        except OSError:
             pass 
        
        # 6. FINAL RESULT OBJECT
        disease_info = get_disease_info(prediction_key)
        
        result = {
            'diagnosis_name': disease_info['name'],
            'confidence': f'{preds[top_index] * 100:.2f}%',
            'gradcam_url': gradcam_base64_data, 
            'channel_weights': channel_weights, 
            'rgb_layers': rgb_layer_data, # NEW
            'info': disease_info
        }
        
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    if model is None:
        print("\n[FLASK STARTUP SKIPPED] Server cannot run without a loaded model.")
    else:
        app.run(debug=True)