# app/services/preprocessing.py

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torchvision import transforms


class ImagePreprocessor:

    def __init__(
        self,
        patch_size=256,
        image_size=1024
    ):

        self.patch_size = patch_size
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_image(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        return image

    def load_numpy(self, image):

        if isinstance(image, Image.Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Image must be PIL Image or NumPy array."
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected HWC image, got {image.shape}"
            )

        return image

    def resize(self, image):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        return image

    def image_to_tensor(self, image):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.resize(image)

        tensor = self.transform(image)

        return tensor

    def extract_patches(self, image):

        """
        Split 1024x1024 image into 16 non-overlapping
        256x256 patches.

        Returns:
            list of:
            {
                "tensor": tensor,
                "x": x,
                "y": y
            }
        """

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.resize(image)

        patches = []

        for y in range(
            0,
            self.image_size,
            self.patch_size
        ):

            for x in range(
                0,
                self.image_size,
                self.patch_size
            ):

                patch = image.crop(
                    (
                        x,
                        y,
                        x + self.patch_size,
                        y + self.patch_size
                    )
                )

                tensor = self.transform(patch)

                patches.append({
                    "tensor": tensor,
                    "x": x,
                    "y": y
                })

        return patches

    def prepare_pair(
        self,
        before_path,
        after_path
    ):

        before = self.load_image(before_path)
        after = self.load_image(after_path)

        before = self.resize(before)
        after = self.resize(after)

        return before, after