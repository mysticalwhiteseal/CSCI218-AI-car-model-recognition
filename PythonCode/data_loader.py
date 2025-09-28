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
from torchvision.transforms.v2 import ToTensor, ToPILImage, Resize, CenterCrop, RandomRotation, RandomResizedCrop, RandomAffine, GaussianBlur, ColorJitter
from sklearn.model_selection import train_test_split

TARGET_SIZE = (224, 224)
BBOX_CROP_MARGIN = 16
DATA_SRC_ROOT = "data/car_data/train"
DATA_DST_ROOT = "data-processed/train"

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

def crop_to_bbox(img, annotation):
    height, width = img.shape[:2]
    x1 = max(0, annotation['x1'] - BBOX_CROP_MARGIN)
    y1 = max(0, annotation['y1'] - BBOX_CROP_MARGIN)
    x2 = min(annotation['x2'] + BBOX_CROP_MARGIN, width)
    y2 = min(annotation['y2'] + BBOX_CROP_MARGIN, height)

    return img[y1:y2, x1:x2]

def resize(img, target_size=TARGET_SIZE):
    # While maintaining aspect ratio
    target_w, target_h = target_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = int(w*scale)
    new_h = int(h*scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def preprocess_image(class_names, annotations, src_root=DATA_SRC_ROOT, dst_root=DATA_DST_ROOT):
    for class_id, annotation_group in groupby(iterable=annotations, key=itemgetter("class_id")):
        class_name = class_names[class_id - 1]
        if not os.path.exists(dst_root + f"/{class_name}"):
            os.makedirs(dst_root + f"/{class_name}")

        for annotation in annotation_group:
            src_img = cv2.imread(filename=src_root + f"/{class_name}/{annotation['fname']}")
            cropped_img = crop_to_bbox(src_img, annotation)
            # cv2.imshow('image', cropped_img)
            # cv2.waitKey(0)
            resized_img = resize(cropped_img)
            # cv2.imshow('image', resized_img)
            # cv2.waitKey(0)

            cv2.imwrite(filename=dst_root + f"/{class_name}/{annotation['fname']}", img=resized_img)

def get_names(path):
    reader = csv.reader(open(path))
    names = chain.from_iterable(list(reader))
    names = [f.replace("/", "-") for f in names]
    return names

def get_annotations(path):
    reader = csv.DictReader(f=open(path), fieldnames=["fname","x1","y1","x2","y2","class_id"])
    annotations = []
    for row in reader:
        row['x1'] = int(row['x1'])
        row['y1'] = int(row['y1'])
        row['x2'] = int(row['x2'])
        row['y2'] = int(row['y2'])
        row['class_id'] = int(row['class_id'])
        annotations.append(row)

    return sorted(annotations, key=itemgetter("class_id"))

names = get_names("data/names.csv")
annotations = get_annotations("data/anno_train.csv")

preprocess_image(names, annotations)
