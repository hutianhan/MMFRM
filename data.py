import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import random

def add_gaussian_noise(image_array, mean=0.0, var=30):
     std = var**0.5
     noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
     noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
     return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d



class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, basic_aug=False):
        self.phase = phase
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:,
                     LABEL_COLUMN].values-1   # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)

        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize((32, 32)),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(scale=(0.02, 0.25))])

            image = data_transforms(image)
        else:
            data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            image = data_transforms(image)

        return image, label, idx




