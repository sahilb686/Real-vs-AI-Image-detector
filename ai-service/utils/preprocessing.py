"""
Image preprocessing pipelines for each supported classifier.

Each function mirrors the original Streamlit preprocessing exactly:
same target sizes, normalization, and array shapes. Only the prediction
step has been moved to `services/predictor.py` for separation of concerns.
"""

from typing import BinaryIO

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def preprocess_cnn(image_source: str | BinaryIO) -> np.ndarray:
    """
    Preprocess an image for the CNN classifier.

    Original behaviour preserved:
      - Resize to 256×256
      - Normalize pixel values to [0, 1] by dividing by 255
      - Add batch dimension → shape (1, 256, 256, 3)
    """
    input_picture = load_img(image_source, target_size=(256, 256))
    img_arr = img_to_array(input_picture) / 255.0
    img_arr = img_arr.reshape((1, 256, 256, 3))
    return img_arr


def preprocess_efficientnet(image_source: str | BinaryIO) -> np.ndarray:
    """
    Preprocess an image for the EfficientNetB3 classifier.

    Original behaviour preserved:
      - Resize to 300×300
      - Convert to array (no normalization)
      - Add batch dimension → shape (1, 300, 300, 3)
    """
    img = load_img(image_source, target_size=(300, 300))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def preprocess_efficientnet_art(image_source: str | BinaryIO) -> np.ndarray:
    """
    Preprocess an image for the EfficientNet Art fine-tuned classifier.

    Original behaviour preserved:
      - Resize to 224×224
      - Convert to array (no normalization)
      - Add batch dimension → shape (1, 224, 224, 3)
    """
    img = load_img(image_source, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr
