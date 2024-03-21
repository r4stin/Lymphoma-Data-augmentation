import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
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

##########################
### SETTINGS
##########################

# RANDOM_SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 20
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




# Custom dataset class to include both data and labels
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
#         self.labels = labels
        self.labels = labels
#         self.lb = LabelBinarizer()
#         self.one_hot_labels = self.lb.fit_transform(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels[idx]
#         label = torch.tensor(self.one_hot_labels[idx], dtype=torch.float)
        if self.transform:
            item = self.transform(item)
        return item, label

    
    
    
    


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size, augment=None, do_more_transforms=False, more_transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.augment = augment
        self.do_more_transforms = do_more_transforms
        self.more_transforms = more_transforms

    def setup(self, stage: str):
        data = sio.loadmat('/kaggle/input/lymphoma/DatasColor_29.mat')

        datas = data['DATA'][0][0][0]
        labels = data['DATA'][0][1][0]
        test_size = 0.2
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(datas, labels, test_size=test_size, random_state=42)

        if stage == 'no_augment':            
            self.train_dataset = CustomDataset(self.X_train, self.y_train, transform=AllTransforms.NO_TRANSFORM)
            self.test_dataset = CustomDataset(self.X_test, self.y_test, transform=AllTransforms.NO_TRANSFORM)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [0.8, 0.2])

            print("len train: ", len(self.train_dataset))
            print("len val: ", len(self.val_dataset))
            print("len test: ", len(self.test_dataset))



            

        elif stage == 'do_augment':
            preprocess_dataset = CustomDataset(self.X_train, self.y_train, transform=AllTransforms.PREPROCESS)
            self.test_dataset = CustomDataset(self.X_test, self.y_test, transform=AllTransforms.PREPROCESS)
            
            augmented_dataset = CustomDataset(self.X_train, self.y_train, transform=self.augment)
            
            
            if self.do_more_transforms:
                more_transforms_dataset = CustomDataset(self.X_train, self.y_train, transform=self.more_transforms)
                augmented_dataset = ConcatDataset([augmented_dataset, more_transforms_dataset])
        
            self.train_dataset = ConcatDataset([preprocess_dataset, augmented_dataset])
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [0.8, 0.2])

            print("len train: ", len(self.train_dataset))
            print("len val: ", len(self.val_dataset))
            print("len test: ", len(self.test_dataset))


        else:
            print('Invalid stage')
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    # 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    # 
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
