import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from sklearn.decomposition import PCA
import numpy as np


class APP6:

    def apply_pca_perturbation_2(image):
        image_tensor = transforms.ToTensor()(image)

        # Separate channels
        red_channel = image_tensor[0]
        green_channel = image_tensor[1]
        blue_channel = image_tensor[2]

        channels = [red_channel, green_channel, blue_channel]

        perturbed_channels = []

        for channel in channels:
            # Flatten and get pixel values
            flat_channel = channel.view(-1).cpu().numpy()

            # Compute PCA
            pca = PCA(n_components=1)  # Perform PCA for each channel separately
            pca.fit(flat_channel.reshape(-1, 1))

            eigenvectors = torch.tensor(pca.components_).float()
            eigenvalues = torch.tensor(pca.explained_variance_).float()

            alpha = torch.randn(1) * 0.1

            # Compute perturbation
            perturbation = torch.matmul(eigenvectors.T, alpha * eigenvalues)

            # Add perturbation to channel
            perturbed_channel = flat_channel + perturbation.numpy().flatten()

            # Clip pixel values to [0, 1]
            perturbed_channel = np.clip(perturbed_channel, 0, 1)

            # Reshape to original shape
            perturbed_channel = torch.tensor(perturbed_channel).view_as(channel)
            perturbed_channels.append(perturbed_channel)

        # Combine perturbed channels
        perturbed_image = torch.stack(perturbed_channels, dim=0)

        # Apply resizing and cropping
        perturbed_image = transforms.Compose([
            #         transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.RandomCrop(224),
            #         transforms.ToTensor()
        ])(perturbed_image)

        return perturbed_image

    def apply_pca_perturbation_1(image):
        image_tensor = transforms.ToTensor()(image)

        # Separate channels
        red_channel = image_tensor[0]
        green_channel = image_tensor[1]
        blue_channel = image_tensor[2]

        channels = [red_channel, green_channel, blue_channel]

        perturbed_channels = []

        for channel in channels:
            # Flatten and get pixel values
            flat_channel = channel.view(-1).cpu().numpy()

            # Compute PCA
            pca = PCA(n_components=1)  # Perform PCA for each channel separately
            pca.fit(flat_channel.reshape(-1, 1))

            eigenvectors = torch.tensor(pca.components_).float()
            eigenvalues = torch.tensor(pca.explained_variance_).float()

            alpha = torch.randn(1) * 0.2

            # Compute perturbation
            perturbation = torch.matmul(eigenvectors.T, alpha * eigenvalues)

            # Add perturbation to channel
            perturbed_channel = flat_channel + perturbation.numpy().flatten()

            # Clip pixel values to [0, 1]
            perturbed_channel = np.clip(perturbed_channel, 0, 1)

            # Reshape to original shape
            perturbed_channel = torch.tensor(perturbed_channel).view_as(channel)
            perturbed_channels.append(perturbed_channel)

        # Combine perturbed channels
        perturbed_image = torch.stack(perturbed_channels, dim=0)

        # Apply resizing and cropping
        perturbed_image = transforms.Compose([
            #         transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.RandomCrop(224),
            #         transforms.ToTensor()
        ])(perturbed_image)

        return perturbed_image

    def gaussian_noisy2mul(img):
        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ])(img)
        img = img.permute(1, 2, 0)

        noise = np.random.normal(loc=0, scale=1, size=img.shape)

        noisy2mul = np.clip((img * (1 + noise * 0.2)), 0, 1)

        noisy2mul = noisy2mul.permute(2, 0, 1)

        return noisy2mul

    def gaussian_n2(img):
        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ])(img)
        # img = img[..., ::-1]/255.0
        img = img.permute(1, 2, 0)
        noise = np.random.normal(loc=0, scale=1, size=img.shape)

        img2 = img * 2
        n2 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.2)), (1 - img2 + 1) * (1 + noise * 0.2) * -1 + 2) / 2,
                     0, 1)
        n2 = transforms.ToTensor()(n2)

        return n2

    APP6 = [apply_pca_perturbation_1, gaussian_noisy2mul, gaussian_n2, apply_pca_perturbation_2]

