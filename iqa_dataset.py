import glob
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class IQA_DATASET(Dataset):
    def __init__(self, folder, transform=None, toy=False):
        self.transforms=transform

        self.imgs = glob.glob(folder+'//*')
        if toy:
            self.imgs = self.imgs[:len(imgs)//10]


    def __getitem__(self, index):
        file = self.imgs[index]
        img = Image.open(file)
        if self.transforms:
            img = self.transforms(img)
            
        targetB = int(os.path.split(file)[1].split('_')[2])
        targetB = round(targetB*0.16+0.1,2)
        targetN = int(os.path.split(file)[1].split('_')[4])
        targetN = round(targetN*0.16+0.1,2)

        return img,  torch.tensor([targetB, targetN])

    def __len__(self):
        return len(self.imgs)
