import pandas as pd
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms

# open config
with open('config.json') as config_file:
    config = json.load(config_file)

# preprocessing of the pictures
IMAGE_SIZE = 224                              # Image size (224x224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

def load_and_format_image(path, type, normalization=False):
    """loads images from the dataset resizes them and transforms them if needed"""
    image_transformation = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    ]
    if type == 'train':
        #image_transformation.append(transforms.RandomHorizontalFlip(p=0.5))
        image_transformation.append(transforms.ToTensor())
    else:
        image_transformation.append(transforms.ToTensor())
    if normalization:
        # Normalization with mean and std from ImageNet
        image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    image_transformation = transforms.Compose(image_transformation)
    img = Image.open(path).convert("RGB")                                               # 32,32,3
    img = image_transformation(img)                                                     # 3,32,32
    img_array = np.array(img)
    return img_array

def load_label(value):
    """loads labels from the dataset"""
    labels = []
    if value == 1:
        labels.append(1)
    else:
        labels.append(0)
    return labels

def image_train_gen(path_to_csv, length, type):
    """creates and returns arrays for images and labels"""
    i = 0
    df_train = pd.read_csv(path_to_csv)
    df_train.sample(frac=1)  # shuffle the data frame
    while True:
        X = []
        y = []
        for b in range(length):
            data_point = df_train.iloc[i]
            i += 1
            X.append(load_and_format_image(config["path_to_image_data"] + str(data_point["Path"]), type))
            y.append(load_label(data_point["Pleural Effusion"]))

        return np.array(X), np.array(y)
