"""
Model loader service — singleton pattern.

All TensorFlow models are built and loaded exactly once during FastAPI
startup. The CNN architecture is no longer recreated per request; weights
are loaded a single time and kept in memory for the process lifetime.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

from app.config import (
    CNN_WEIGHTS_PATH,
    EFFICIENTNET_ART_MODEL_PATH,
    EFFICIENTNET_MODEL_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    """Container for all in-memory model references."""

    cnn_model: Sequential
    efficientnet_model: tf.keras.Model
    efficientnet_art_model: tf.keras.Model
    distribution_strategy: tf.distribute.Strategy


class ModelLoader:
    """
    Thread-safe singleton that loads models once at application startup.

    Follows the Singleton pattern: only one instance exists per process,
    and `load_all` is idempotent.
    """

    _instance: Optional["ModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = None
                    cls._instance._initialized = False
        return cls._instance

    @property
    def models(self) -> LoadedModels:
        """Return loaded models; raises if startup loading has not completed."""
        if self._models is None:
            raise RuntimeError("Models have not been loaded. Call load_all() first.")
        return self._models

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    def load_all(self) -> LoadedModels:
        """
        Load every model into memory. Safe to call multiple times;
        subsequent calls return the cached models without reloading.
        """
        if self._initialized and self._models is not None:
            logger.info("Models already loaded — skipping reload.")
            return self._models

        with self._lock:
            if self._initialized and self._models is not None:
                return self._models

            logger.info("Loading models into memory (one-time startup)...")
            self._validate_model_files()
            strategy = self._create_distribution_strategy()

            cnn_model = self._build_and_load_cnn()
            efficientnet_model = self._load_keras_model(EFFICIENTNET_MODEL_PATH, "EfficientNetB3")
            efficientnet_art_model = self._load_keras_model(
                EFFICIENTNET_ART_MODEL_PATH, "EfficientNet Art"
            )

            self._models = LoadedModels(
                cnn_model=cnn_model,
                efficientnet_model=efficientnet_model,
                efficientnet_art_model=efficientnet_art_model,
                distribution_strategy=strategy,
            )
            self._initialized = True
            logger.info("All models loaded successfully.")
            return self._models

    def _validate_model_files(self) -> None:
        """Ensure all weight files exist before attempting TensorFlow loads."""
        required_files = [
            CNN_WEIGHTS_PATH,
            EFFICIENTNET_MODEL_PATH,
            EFFICIENTNET_ART_MODEL_PATH,
        ]
        missing = [str(path) for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model file(s): {', '.join(missing)}. "
                "Place .h5 weights in the ai-service/models/ directory."
            )

    @staticmethod
    def _create_distribution_strategy() -> tf.distribute.Strategy:
        """
        Create a TPU distribution strategy when available, otherwise default.

        Mirrors the original Streamlit try/except TPU logic used by
        EfficientNet inference functions.
        """
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            logger.info("TPU distribution strategy initialized.")
        except ValueError:
            strategy = tf.distribute.get_strategy()
            logger.info("Default distribution strategy in use (no TPU detected).")
        return strategy

    @staticmethod
    def _build_and_load_cnn() -> Sequential:
        """
        Build the CNN architecture once and load pre-trained weights.

        Architecture matches the original Streamlit app exactly so
        prediction behaviour is unchanged.
        """
        logger.info("Building CNN architecture and loading weights from %s", CNN_WEIGHTS_PATH)

        model = Sequential()
        model.add(
            Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                input_shape=(256, 256, 3),
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.09))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.load_weights(str(CNN_WEIGHTS_PATH))

        logger.info("CNN model ready.")
        return model

    @staticmethod
    def _load_keras_model(path: Path, label: str) -> tf.keras.Model:
        """Load a full Keras .h5 model from disk."""
        logger.info("Loading %s from %s", label, path)
        model = tf.keras.models.load_model(str(path))
        logger.info("%s loaded.", label)
        return model


# Module-level singleton accessor used by predictor and startup hook.
model_loader = ModelLoader()
