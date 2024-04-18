import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class APP1:

    # Define custom data augmentation functions
    def random_horizontal_flip(image):
        return TF.hflip(image)

    def random_vertical_flip(image):
        return TF.vflip(image)

    def random_resized_crop(image, scale_range=(1, 1.2), size=(224, 224)):
        width, height = image.size

        # Randomly select a scale factor
        scale_factor = random.uniform(*scale_range)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Ensure new dimensions are not larger than the original dimensions
        new_width = min(new_width, width)
        new_height = min(new_height, height)

        # Randomly select top-left corner for cropping
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        # Crop and resize
        image = TF.crop(image, top, left, new_height, new_width)
        image = TF.resize(image, size)

        return image
    APP1 = [
            random_horizontal_flip,
            random_vertical_flip,
            random_resized_crop
        ]
    
