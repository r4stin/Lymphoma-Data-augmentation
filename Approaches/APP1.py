import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class APP1:

    def random_horizontal_flip(image):
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(224),
          
        ])(image)
        return image

    def random_vertical_flip(image):
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomCrop(224),
        ])(image)
        return image
    
    def random_affine(image):
        
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=(-10, 10), translate=(0, 0.05), shear=(0, 30), scale=(1, 1.2)),
            transforms.CenterCrop(224),
        ])(image)
        return image

    APP1 = [
            random_horizontal_flip,
            random_vertical_flip,
            random_affine
        ]


def __getattr__():
    return APP1.APP1
    
