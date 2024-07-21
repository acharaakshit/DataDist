from PIL import Image
import argparse
import numpy as np
import os
import yaml
from tqdm import tqdm
import models
import utils
import pandas as pd
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import adsam_models

def save_outputs(folder, feature_extractor, checkpoint, size):
    images = []
    for filename in os.listdir(folder):
        images.append(os.path.join(folder, filename))
    
    all_outputs = []

    if feature_extractor == "EfficientNet":
        model = models.EfficientNetB3FeatureExtractor()
    elif feature_extractor == "DinoV2":
        model = models.DinoV2FeatureExtractor(modelpath=checkpoint)
    elif feature_extractor == "CLIPSegForImageSegmentation":
        model = models.CLIPSegForImageSegmentationFeatureExtractor(modelpath=checkpoint)
    elif feature_extractor == "CLIPSegModel":
        model = models.CLIPSegModelFeatureExtractor(modelpath=checkpoint)
    elif feature_extractor == "SAM":
        model = models.SAMFeatureExtractor(modelpath=checkpoint)
    elif feature_extractor == "Unet++":
        model = models.UnetPPModel("unetplusplus", "efficientnet-b3", in_channels=3, out_classes=1)
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    elif feature_extractor == "DeepCrack":
        model = models.DeepCrack()
        trainer = models.DeepCrackTrainer(model)
        model.load_state_dict(trainer.saver.load(checkpoint))
    elif feature_extractor == "SAMAdapter":
        with torch.no_grad():
            config = './config-sam-adapter.yaml'
            with open(config, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            model = adsam_models.make(config['model'])
            sam_checkpoint = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(sam_checkpoint, strict=True)
        # change input image size to required size of 1024*1024
        size = (1024, 1024)
        print(f"Image shape changed to {size}")
    # add your custom model here
    
    img_names = []
    for image_path in tqdm(images):
        img = Image.open(image_path).convert(mode="RGB")
        img_names.append(image_path)
        img.thumbnail(size, Image.LANCZOS)
        img = img.resize(size, Image.LANCZOS)
        img = np.array(img.convert('RGB'))

        if feature_extractor == "EfficientNet":
            output = model.extract_features(np.array([img]))
        elif feature_extractor == "DinoV2":
            output = model.extract_features(np.array([img]))
        elif feature_extractor == "CLIPSegForImageSegmentation":
            prompt = ["line structures"]
            output = model.extract_features(img, prompt=prompt)
        elif feature_extractor == "CLIPSegModel":
            output = model.extract_features(img)
        elif feature_extractor == "SAM":
            output = model.extract_features(img_array=img)
        elif feature_extractor == "Unet++":
            img = np.expand_dims(np.transpose(img, (2,1,0)), axis=0)
            img = torch.from_numpy(img)
            output = model.model.encoder(img.float())
            output = torch.cat([tensor.reshape(-1) for tensor in output])
        elif feature_extractor == "DeepCrack":
            img = np.array([np.array(img).transpose(2,0,1).astype(np.float32) / 255])
            return_nodes = {}
            for layer in range(1,6):
                return_nodes[f"down{layer}"] = f"encoder{layer}"
            deepcrack_model = create_feature_extractor(model, return_nodes=return_nodes)
            img = torch.from_numpy(img).float()
            features = list(deepcrack_model(img).values())
            output = torch.cat([tensor.reshape(-1) for tensor in features])
        elif feature_extractor == "SAMAdapter":
            img = np.array([np.array(img).transpose(2,0,1).astype(np.float32)/255])
            img = torch.from_numpy(img)
            output = model.image_encoder(img.float())
        # add custom prediction function suitable for your model
        
        try:
            # if using GPU
            all_outputs.append(np.array(output.ravel().detach().cpu().numpy()))
        except Exception as e:
            all_outputs.append(np.array(output.ravel()))

    return all_outputs, img_names

def main():
    parser = argparse.ArgumentParser()
    # number of dimensions
    parser.add_argument('--n', default=26)
    parser.add_argument('--model', default='EfficientNet')
    parser.add_argument('--dataconfig', default='data_config.yaml')
    parser.add_argument('--modelconfig', default='model_config.yaml')
    args = parser.parse_args()

    # load the model checkpoints/weights
    with open(args.modelconfig, "r") as yamlfile:
        checkpoints = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    if args.model not in checkpoints.keys():
        raise ValueError("Invalid value passed for model/feature extractor")

    n_components = args.n

    prefix = "./data/"
    # setup the primary and secondary datasets
    with open(args.dataconfig, "r") as yamlfile:
        categories = yaml.load(yamlfile, Loader=yaml.FullLoader)

    primary_path = os.path.join(prefix, categories['primary'])
    secondary_paths = []
    for s in categories['secondary']:
        print(s)
        secondary_paths.append(os.path.join(prefix, s))
    
    # image size
    h = categories['height']
    w = categories['width']
    
    primary_outputs, primary_names = save_outputs(folder=primary_path, feature_extractor=args.model, checkpoint=checkpoints[args.model], size=(h,w))

    df_lists = []

    for secondary_path, secondary_dataset in zip(secondary_paths, categories['secondary']):
        secondary_outputs, secondary_names = save_outputs(folder=secondary_path, feature_extractor=args.model, checkpoint=checkpoints[args.model], size=(h,w))
        scores = utils.reduce_dimensions(outputs=primary_outputs+secondary_outputs,
                                        names=(primary_names,secondary_names),
                                        n_components=n_components,
                                        feature_extractor=args.model,
                                        dataset_name=secondary_dataset)
        print(scores)
        df_lists.append(scores.values())
    
    df = pd.DataFrame(df_lists, columns=list(scores.keys()), index=categories['secondary'])
    df.update(df.div(df.sum(axis=0),axis=1))
    print(df)


if __name__=="__main__":
    main()