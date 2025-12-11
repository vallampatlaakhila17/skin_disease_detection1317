# src/model_architecture.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation, multiply, Reshape, MaxPooling2D
from tensorflow.keras.models import Model

# === ATTENTION MECHANISM: FOCUS & FILTER ===
def channel_attention(input_feature, ratio=8):
    """Implements a Squeeze-and-Excitation style Channel Attention Module."""
    channel = input_feature.shape[-1]
    
    # Squeeze: Global Average Pooling
    squeeze = GlobalAveragePooling2D()(input_feature)

    # Excitation: Two fully connected layers
    excitation = Dense(channel // ratio, activation='relu', use_bias=False)(squeeze)
    excitation = Dense(channel, activation='sigmoid', use_bias=False)(excitation)
    
    # Scale: Reshape and multiply with input feature
    excitation = Reshape((1, 1, channel))(excitation)
    scaled_feature = multiply([input_feature, excitation])
    
    return scaled_feature

# === CNN ARCHITECTURE ===
def create_attention_cnn_model(input_shape=(128, 128, 3), num_classes=7):
    """Builds the CNN integrated with the Attention block."""
    input_layer = tf.keras.Input(shape=input_shape)
    
    # Block 1 - Layer named 'conv2d' automatically
    x = Conv2D(64, (3, 3), padding='same')(input_layer) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2 - Layer named 'conv2d_1' automatically (Target for Grad-CAM)
    x = Conv2D(128, (3, 3), padding='same')(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # ATTENTION INTEGRATION POINT
    x = channel_attention(x) 
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    # Classification Head
    x = GlobalAveragePooling2D()(x) 
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model