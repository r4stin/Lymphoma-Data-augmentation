import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from scipy.fftpack import dct, idct  # import for discrete cosine transform


class APP4:
    def DCT_method1(image):
        y, x, z = image.shape
        # Create a blank image
        image = np.zeros_like(image, dtype=np.uint8)
        # Loop over each channel
        for channel in range(z):
            # Apply DCT
            DCTImage = dct(dct(image[:, :, channel], axis=0, norm='ortho'), axis=1, norm='ortho')
            d1 = DCTImage.copy()  # a copy for unmodified pixel in position (1,1)

            zero_indices = np.random.choice([False, True], size=DCTImage.shape,
                                            p=[0.9, 0.1])  # 0.6 probability of TRUE and 0.4 probability of FALSE.

            sigma = np.std(DCTImage[zero_indices])
            sigma = sigma / 2
            random_z = np.random.uniform(-0.5, 0.5) * sigma

            DCTImage[zero_indices] += random_z

            DCTImage[0, 0] = d1[0, 0]
            # Inverse DCT
            image[:, :, channel] = idct(idct(DCTImage, axis=0, norm='ortho'), axis=1, norm='ortho')
        image.astype(np.uint8)
        image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),

        ])(image)
        return image

    APP4 = [DCT_method1]

