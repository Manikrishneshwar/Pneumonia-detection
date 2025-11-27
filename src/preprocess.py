# src/preprocess.py

import numpy as np
from tensorflow.keras.preprocessing import image

def load_image(img_path, target_size=(224, 224)):
    """
    Loads and preprocesses an image for model inference.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    img_array = img_array / 255.0  # normalize
    return img_array

def load_test_images(test_dir, target_size=(224, 224)):
    """
    Loads all images from test directory into arrays.
    
    test_dir contains:
      NORMAL/
      PNEUMONIA/
    """
    import os

    images = []
    labels = []

    classes = ["NORMAL", "PNEUMONIA"]

    for label in classes:
        class_dir = os.path.join(test_dir, label)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, fname)
                img = load_image(img_path, target_size)
                images.append(img)
                labels.append(0 if label == "NORMAL" else 1)

    images = np.vstack(images)
    labels = np.array(labels)

    return images, labels
