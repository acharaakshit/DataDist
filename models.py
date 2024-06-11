import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from transformers import AutoImageProcessor, Dinov2Model, CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from segment_anything import sam_model_registry, SamPredictor

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

class CLIPSegForImageSegmentationFeatureExtractor:
    def __init__(self, modelpath):
        self.image_processor = CLIPSegProcessor.from_pretrained(modelpath, local_files_only=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained(modelpath, local_files_only=True)

    def extract_features(self, img_array, prompt):
        inputs = self.image_processor(text=prompt, images=[img_array]*len(prompt), padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        output = torch.stack(outputs.decoder_output.hidden_states)
        return output

class SAMFeatureExtractor:
    def __init__(self, modelpath):
        self.model = sam_model_registry['vit_b'](checkpoint=modelpath)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.predictor = SamPredictor(self.model)

    def extract_features(self, img_array):
        self.predictor.set_image(img_array)
        output = self.predictor.get_image_embedding()
        return output