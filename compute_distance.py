from sys import prefix
from PIL import Image
import argparse
import numpy as np
import os
import yaml
from tqdm import tqdm
import models
import utils
import pandas as pd

def save_outputs(folder, feature_extractor, checkpoint, size):
    images = []
    for filename in os.listdir(folder):
        images.append(os.path.join(folder, filename))
    
    all_outputs = []

    if feature_extractor == "EfficientNet":
        model = models.EfficientNetB3FeatureExtractor()
    # add your custom model here
    
    img_names = []
    for image_path in tqdm(images):
        img = Image.open(image_path).convert(mode="RGB")
        img_names.append(image_path)
        img.thumbnail(size, Image.LANCZOS)
        img.resize(size, Image.LANCZOS)
        img = np.array(img.convert('RGB'))

        if feature_extractor == "EfficientNet":
            output = model.extract_features(np.array([img]))
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