"""Infrastructure-defect image classifier (MobileNetV2, see ml/cv/train.py)."""

import json
import logging
import threading

import numpy as np

from app.config import get_settings
from ml.cv.preprocess import ImageRejected, load_and_prepare

log = logging.getLogger(__name__)


class Classifier:
    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            settings = get_settings()
            self._loaded = True
            if not settings.cv_model_path.exists():
                log.warning("CV model not found at %s; classification disabled", settings.cv_model_path)
                return
            import tensorflow as tf  # heavy import, deferred until first use

            self._model = tf.keras.models.load_model(settings.cv_model_path)
            self._labels = json.loads(settings.cv_labels_path.read_text())["classes"]
            log.info("Loaded CV model with %d classes", len(self._labels))

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def predict(self, img_bytes: bytes) -> dict:
        """Return {prediction, confidence, top3} or nulls if unavailable/rejected."""
        empty = {"prediction": None, "confidence": None, "top3": []}
        if not self.available:
            return empty
        try:
            img = load_and_prepare(img_bytes)
        except ImageRejected as e:
            log.info("Image rejected: %s", e)
            return empty
        probs = self._model.predict(np.expand_dims(img, 0), verbose=0)[0]
        order = np.argsort(probs)[::-1]
        return {
            "prediction": self._labels[order[0]],
            "confidence": float(probs[order[0]]),
            "top3": [{"label": self._labels[i], "p": float(probs[i])} for i in order[:3]],
        }


classifier = Classifier()
