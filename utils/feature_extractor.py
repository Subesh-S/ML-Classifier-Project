import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings

# Reduce noisy warnings from TensorFlow / Keras / numpy
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2 # type: ignore
import numpy as np# type: ignore
import tensorflow as tf# type: ignore
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input# type: ignore

# Ensure TensorFlow logger is quiet
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load pretrained CNN once
feature_extractor = MobileNetV2(
    
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

CLASSES = ["electric bus", "electric car"]

def extract_features(folder):
    features, labels = [], []
    
    # Collect all images first, then batch process
    image_data = []
    image_labels = []

    for label, cls in enumerate(CLASSES):
        path = os.path.join(folder, cls)
        if not os.path.isdir(path):
            print(f"Warning: Class path not found: {path}")
            continue

        files = os.listdir(path)
        print(f"Processing class '{cls}': found {len(files)} files")
        
        for file in files:
            img_path = os.path.join(path, file)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: Could not read image: {img_path}")
                continue

            # OpenCV loads images in BGR; convert to RGB expected by preprocess_input
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                continue

            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32)
            img = preprocess_input(img)
            
            image_data.append(img)
            image_labels.append(label)

    if len(image_data) == 0:
        print(f"ERROR: No images were successfully processed from {folder}")
        print(f"Please check that the path exists and contains image files in subdirectories: {CLASSES}")
        return np.array(features), np.array(labels)
    
    # Batch predict all images at once (much faster than one-by-one)
    print(f"  Extracting features from {len(image_data)} images using batch prediction...")
    batch_data = np.array(image_data)
    try:
        batch_features = feature_extractor.predict(batch_data, verbose=0)
        for feature in batch_features:
            features.append(feature.flatten())
        labels = image_labels
    except Exception as e:
        print(f"ERROR during feature extraction: {e}")
        return np.array(features), np.array(labels)
    
    print(f"  Successfully extracted features from {len(features)} images")
    
    return np.array(features), np.array(labels)
