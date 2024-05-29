import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np


class APP2:

    def gaussian_noisy_1(img):
        img = img.astype(np.float32) / 255.0

        noise = np.random.normal(loc=0, scale=1, size=img.shape)

        noisy_img = np.clip((img * (1 + noise * 0.2)), 0, 1)

        noisy_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(224),
        ])(noisy_img)

        return noisy_img.float()

    def gaussian_noisy_2(img):
        img = img.astype(np.float32) / 255.0

        # Inject noise into the original ndarray image
        noise = np.random.normal(loc=0, scale=1, size=img.shape)

        img2 = img * 2
        noisy_img = np.clip(
            np.where(img2 <= 1, (img2 * (1 + noise * 0.2)), (1 - img2 + 1) * (1 + noise * 0.2) * -1 + 2) / 2,
            0, 1)

        noisy_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(224),
        ])(noisy_img)

        return noisy_img.float()

    APP2 = [gaussian_noisy_1, gaussian_noisy_2]


def __getattr__():
    return APP2.APP2
