import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from transformers import AutoImageProcessor, Dinov2Model
import torch

class EfficientNetB3FeatureExtractor:
    def __init__(self):
        self.model = EfficientNetB3(weights='imagenet', include_top=False)

    def extract_features(self, img_array):
        features = self.model.predict(img_array)
        return features

class DinoV2FeatureExtractor:
    def __init__(self, modelpath):
        self.image_processor = AutoImageProcessor.from_pretrained(modelpath)
        self.model = Dinov2Model.from_pretrained(modelpath)

    def extract_features(self, img_array):
        inputs = self.image_processor(img_array, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        output = outputs.last_hidden_state
        return output