import torch
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import numpy as np
import random


class APP3:

    def apply_pca_perturbation(image):
        # Convert the image to a NumPy array and ensure it is in the range [0, 1]
        image = np.clip(image.astype(np.float32) / 255.0, 0, 1)

        # Separate the image into its color channels
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]

        channels = [red_channel, green_channel, blue_channel]
        perturbed_channels = []

        # Randomly choose between 0.1 and 0.2 for alpha
        alpha_value = random.choice([0.1, 0.2])

        for channel in channels:
            # Flatten the channel to get pixel values
            flat_channel = channel.flatten()

            # Compute PCA for the flattened channel
            pca = PCA(n_components=1)
            pca.fit(flat_channel.reshape(-1, 1))

            eigenvectors = torch.tensor(pca.components_).float()
            eigenvalues = torch.tensor(pca.explained_variance_).float()

            alpha = torch.randn(1) * alpha_value

            # Compute the perturbation
            perturbation = torch.matmul(eigenvectors.T, alpha * eigenvalues)

            # Add the perturbation to the channel
            perturbed_channel = flat_channel + perturbation.numpy().flatten()

            # Clip pixel values to the range [0, 1]
            perturbed_channel = np.clip(perturbed_channel, 0, 1)

            # Reshape to the original channel shape
            perturbed_channel = perturbed_channel.reshape(channel.shape)
            perturbed_channels.append(perturbed_channel)

        # Combine the perturbed channels back into a single image
        perturbed_image = np.stack(perturbed_channels, axis=-1)

        # Convert the perturbed image to a tensor
        perturbed_image = transforms.ToTensor()(perturbed_image)

        # Apply random crop to the perturbed image
        perturbed_image = transforms.RandomCrop(224)(perturbed_image)

        return perturbed_image

    APP3 = [apply_pca_perturbation]


def __getattr__():
    return APP3.APP3


