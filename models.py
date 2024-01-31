import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB3

class EfficientNetB3FeatureExtractor:
    def __init__(self):
        self.model = EfficientNetB3(weights='imagenet', include_top=False)

    def extract_features(self, img_array):
        features = self.model.predict(img_array)
        return features