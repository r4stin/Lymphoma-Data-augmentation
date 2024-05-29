import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from scipy.fftpack import dct, idct  # import for discrete cosine transform


class APP4:

    def apply_dct(im):
        y, x, z = im.shape

        image = np.zeros_like(im, dtype=np.uint8)

        for channel in range(z):
            # Apply DCT
            DCTim = dct(dct(im[:, :, channel], axis=0, norm='ortho'), axis=1, norm='ortho')
            d1 = DCTim.copy()  # a copy for unmodified pixel in position (1,1)

            zero_indices = np.random.choice([False, True], size=DCTim.shape,
                                            p=[0.9, 0.1])  # 0.6 probability of TRUE and 0.4 probability of FALSE.

            sigma = np.std(DCTim[zero_indices])
            sigma = sigma / 2
            random_z = np.random.uniform(-0.5, 0.5) * sigma

            # Modify DCT image
            DCTim[zero_indices] += random_z
            # preserve the unmodified pixel at position (1,1)
            DCTim[0, 0] = d1[0, 0]
            # Inverse DCT
            image[:, :, channel] = idct(idct(DCTim, axis=0, norm='ortho'), axis=1, norm='ortho')

        image.astype(np.uint8)
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(224),
        ])(image)
        return image

    APP4 = [apply_dct]


def __getattr__():
    return APP4.APP4
