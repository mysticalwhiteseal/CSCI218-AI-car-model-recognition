from PIL import Image
from itertools import chain, groupby
from operator import itemgetter
import cv2
import os
import csv
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import ToTensor, ToPILImage, Resize
from sklearn.model_selection import train_test_split

class CarsSubset(Dataset):
    def __init__(self, subset, indices, transform=None):
        self.subset = subset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.subset)

def preprocess_image(class_names, src_root, dst_root, annotations, margin=16):
    for class_id, annotation_group in groupby(iterable=annotations, key=itemgetter("class_id")):
        class_name = class_names[class_id - 1]
        if not os.path.exists(dst_root + f"/{class_name}"):
            os.makedirs(dst_root + f"/{class_name}")

        for annotation in annotation_group:
            src_img = cv2.imread(filename=src_root + f"/{class_name}/{annotation['fname']}")
            height, width = src_img.shape[:2]
            x1 = max(0, annotation['x1'] - margin)
            y1 = max(0, annotation['y1'] - margin)
            x2 = min(annotation['x2'] + margin, width)
            y2 = min(annotation['y2'] + margin, height)

            cropped_img = src_img[y1:y2, x1:x2]
            # cv2.imshow('image', dst_img)
            # cv2.waitKey(0)


            cv2.imwrite(filename=dst_root + f"/{class_name}/{annotation['fname']}", img=cropped_img)

def get_names(path):
    reader = csv.reader(open(path))
    names = chain.from_iterable(list(reader))
    names = [f.replace("/", "-") for f in names]
    return names


names = get_names("data/names.csv")

reader = csv.DictReader(f=open("data\\anno_train.csv"), fieldnames=["fname","x1","y1","x2","y2","class_id"])
annotations = []
for row in reader:
    row['x1'] = int(row['x1'])
    row['y1'] = int(row['y1'])
    row['x2'] = int(row['x2'])
    row['y2'] = int(row['y2'])
    row['class_id'] = int(row['class_id'])
    annotations.append(row)

annotations.sort(key=itemgetter("class_id"))
src_root = "data/car_data/train"
dst_root = "data-processed/train"
margin = 16

preprocess_image(names, src_root, dst_root, annotations, margin)
