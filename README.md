# Skin Disease Classifier and Visualization (pottulu)

## 🌟 Project Overview

This is a full-stack deep learning web application designed for classifying seven common types of skin lesions (e.g., Melanoma, Basal Cell Carcinoma). The application uses a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras and focuses on **Explainable AI (XAI)** by providing visual proof for its predictions.

### Key Features

* **Custom CNN Model:** Uses a high-accuracy Keras model (`.h5`) trained on the HAM10000 dataset for 7-class classification.
* **Explainable AI (XAI) via Grad-CAM:** Implements **Gradient-weighted Class Activation Mapping** to generate a heatmap overlay, showing exactly which regions of the lesion the model focused on for its prediction. 
* **Channel Attention Visualization:** Extracts and normalizes channel weights from a custom attention layer (`channel_attention`) to visualize which features (filters) contributed most to the diagnosis.
* **Full-Stack Interface:** A Flask application handling image uploads, real-time inference, and visualization rendering.
* **Production Ready:** Configured for seamless deployment on cloud platforms like Render using Gunicorn.

---

## 🛠️ Technology Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | Flask (Python) | Routing, handling web requests, and serving dynamic templates. |
| **Deep Learning** | TensorFlow, Keras | Core library for model loading, prediction, and Grad-CAM generation. |
| **Image Processing** | OpenCV (`cv2`), Pillow | Preprocessing, resizing, and generating image overlays for the web. |
| **Server** | Gunicorn | Production-ready WSGI server used for Render deployment. |

---

🔬 Disease Information

| Code | Disease Name | Description |
| :--- | :--- | :--- |
| **mel** | Melanoma | Most dangerous form of skin cancer; uses ABCDE rules for detection. |
| **bcc** | Basal Cell Carcinoma | Most common form; often appears as a waxy or pearly bump. |
| **akiec** | Actinic Keratoses | Pre-cancerous rough, scaly patches in sun-damaged areas. |
| **nv** | Melanocytic Nevus | Common Mole, a typically benign melanocyte tumor. |
| **bkl** | Benign Keratosis-like | Harmless, waxy, "stuck-on" growths (e.g., Seborrheic Keratosis). |
| **df** | Dermatofibroma | Common, firm, benign skin growth often resulting from minor trauma. |
| **vasc** | Vascular Lesions | Benign lesions related to blood vessels (e.g., Angiomas). |

---

