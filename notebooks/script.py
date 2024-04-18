import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import torch
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import pytorch_lightning as L
from torch.utils.data import random_split, DataLoader
from transforms_list import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


##########################
### SETTINGS
##########################

# RANDOM_SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 20



class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=[], train=False):
        self.images = images
        self.labels = labels
        self.augmentations = transform
        self.train = train

    def __len__(self):
        if self.train == True:
        # Total size includes original images and augmented images
            return len(self.images) + len(self.images) * len(self.augmentations)
        else:
            return len(self.images)
    def __getitem__(self, idx):
        if self.train == True:
            if idx <= len(self.images):
                # Original image
                image_pil = TF.to_pil_image(self.images[idx])

                image = TF.resize(image_pil, (224, 224))

                image = TF.to_tensor(image)

                label = self.labels[idx] - 1
            else:
                # Augmented image
                original_idx = idx - len(self.images)
                image = self.images[original_idx % len(self.images)]
                label = self.labels[original_idx % len(self.images)] - 1

                # Apply random augmentation
                augmentation = self.augmentations[original_idx // len(self.images)]
                image_pil = TF.to_pil_image(image)
                image_pil = augmentation(image_pil)
                image = TF.resize(image_pil, (224, 224))

                image = TF.to_tensor(image)
        else:
            image_pil = TF.to_pil_image(self.images[idx])

            image = TF.resize(image_pil, (224, 224))

            image = TF.to_tensor(image)

            label = self.labels[idx] - 1



        return image, label

class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size, transform=[]):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

#     def setup(self, stage: str):
    def setup(self):

        data = sio.loadmat('/kaggle/input/lymphoma/DatasColor_29.mat')

        datas = data['DATA'][0][0][0]
        labels = data['DATA'][0][1][0]
        test_size = 0.2
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(datas, labels, test_size=test_size,
                                                                                random_state=42)
        
        
        self.train_dataset = CustomDataset(self.X_train, self.y_train, transform=self.transform, train=True)
        self.test_dataset = CustomDataset(self.X_test, self.y_test, transform=self.transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [0.8, 0.2])

        print("len train: ", len(self.train_dataset))
        print("len val: ", len(self.val_dataset))
        print("len test: ", len(self.test_dataset))

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    #
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    #
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

