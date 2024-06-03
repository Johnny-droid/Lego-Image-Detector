import os
import json
import sys
import cv2
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import models, transforms
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = ''

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




# ======================== Load Images ========================

def getAllImages(img_files):
    imgs = []

    for img_file in img_files:
        img = loadImage(img_file)
        imgs.append(img)

    return imgs

def loadImage(num, imgLoadSizeRatio = 1, standardSize = -1):
    img = cv2.imread(os.path.join(DATA_DIR, f'{num}'))
    if img is None:
        raise FileNotFoundError(f"Image file '{num}' not found.")
    if standardSize > 0:
        img = cv2.resize(img, (standardSize, standardSize))
    elif imgLoadSizeRatio != 1:
        img = cv2.resize(img, (0, 0), fx = imgLoadSizeRatio, fy = imgLoadSizeRatio)
    
    img = cv2.resize(img, (224, 224))

    img = data_transform(img)
    return img

# ======================== Load Model ========================

def loadModel(model_path = 'model.pth'):
    model = ResNet18Classifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model

class ResNet18Classifier(nn.Module):
    def __init__(self):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 32)

    def forward(self, x):
        return self.resnet(x)
    

# ======================== Make Predictions ========================
def makePredictions(model, imgs):
    model.eval()
    imgs = torch.stack(imgs).to(device)
    preds = []
    
    with torch.no_grad():
        for img in tqdm(imgs):
            img = img.unsqueeze(0)
            pred = model(img)
            prob = F.softmax(pred, dim=1)
            final_pred = torch.argmax(prob, dim=1)
            preds.append(final_pred.cpu().numpy()[0])
    
    for i in range(len(preds)):
        preds[i] += 1

    return preds
        

# ======================== API Input & Output ========================

def inputAPI(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            if 'image_files' in data:
                return data['image_files']
            else:
                print("No 'image_files' key found in the input JSON.")
                return []
    except FileNotFoundError:
        print("File not found:", input_file)
        return []


def outputAPI(items):
    results = []
    for item in items:

        result_item = {
            "file_name": item[0],
            "num_detections": int(item[1])
        }

        results.append(result_item)
    
    output_data = {"results": results}
    
    with open('output.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    

# ======================== Main ========================

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py input.json")
        return

    input_file = sys.argv[1]
    image_files = inputAPI(input_file)
    print("\nList of image files:", image_files)

    # Get all images
    print("\nLoading images...")
    imgs = getAllImages(image_files)
    print("Images loaded.")

    # Load the model
    print("\nLoading model...")
    model = loadModel('resnet18-classifier.pth')
    print("Model loaded.")

    # Make predictions
    print("\nMaking predictions...")
    preds = makePredictions(model, imgs)
    print("Predictions made.")

    # Output the results
    print("\nWriting output to output.json")
    outputAPI(zip(image_files, preds))
    print("\nDone.\n")
        

if __name__ == '__main__':
    main()

