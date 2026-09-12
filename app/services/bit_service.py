import gc
import numpy as np
import torch

from app.services.preprocessing import ImagePreprocessor
from app.models.bit_model import BITModel


class BITService:

    def __init__(
        self,
        checkpoint_path: str,
        bit_repo_path: str = None,
        patch_size: int = 256,
        image_size: int = 1024,
        device: str = None,
    ):

        print("Creating BIT model...")

        self.preprocessor = ImagePreprocessor(
            patch_size=patch_size,
            image_size=image_size,
        )

        self.model = BITModel(
            checkpoint_path=checkpoint_path,
            bit_repo_path=bit_repo_path,
            device=device,
        )

        self.patch_size = patch_size
        self.image_size = image_size

        print("BITService initialized successfully.")

    def detect(
        self,
        before_path: str,
        after_path: str,
    ):

        print("Starting BIT change detection...")

        # --------------------------------------------------
        # Load images
        # --------------------------------------------------

        before_image = self.preprocessor.load_image(
            before_path
        )

        after_image = self.preprocessor.load_image(
            after_path
        )

        # --------------------------------------------------
        # Resize once
        # --------------------------------------------------

        before_image = self.preprocessor.resize(
            before_image
        )

        after_image = self.preprocessor.resize(
            after_image
        )

        # --------------------------------------------------
        # Prepare output mask
        # --------------------------------------------------

        full_mask = np.zeros(
            (
                self.image_size,
                self.image_size
            ),
            dtype=np.uint8
        )

        changed_pixels = 0

        # --------------------------------------------------
        # Process ONE patch at a time
        # --------------------------------------------------

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

                print(
                    f"Processing patch "
                    f"x={x}, y={y}"
                )

                before_patch = before_image.crop(
                    (
                        x,
                        y,
                        x + self.patch_size,
                        y + self.patch_size
                    )
                )

                after_patch = after_image.crop(
                    (
                        x,
                        y,
                        x + self.patch_size,
                        y + self.patch_size
                    )
                )

                # --------------------------------------------------
                # Convert only this patch to tensors
                # --------------------------------------------------

                before_tensor = (
                    self.preprocessor.transform(
                        before_patch
                    )
                    .unsqueeze(0)
                )

                after_tensor = (
                    self.preprocessor.transform(
                        after_patch
                    )
                    .unsqueeze(0)
                )

                # --------------------------------------------------
                # BIT inference
                # --------------------------------------------------

                with torch.no_grad():

                    prediction = (
                        self.model.predict_mask(
                            before_tensor,
                            after_tensor
                        )
                    )

                # --------------------------------------------------
                # Move result immediately to CPU
                # --------------------------------------------------

                patch_mask = (
                    prediction
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                # --------------------------------------------------
                # Reconstruct full mask
                # --------------------------------------------------

                full_mask[
                    y:y + self.patch_size,
                    x:x + self.patch_size
                ] = patch_mask

                changed_pixels += int(
                    np.sum(patch_mask > 0)
                )

                # --------------------------------------------------
                # Explicitly release patch tensors
                # --------------------------------------------------

                del (
                    before_patch,
                    after_patch,
                    before_tensor,
                    after_tensor,
                    prediction,
                    patch_mask,
                )

                gc.collect()

        # --------------------------------------------------
        # Final statistics
        # --------------------------------------------------

        total_pixels = (
            self.image_size *
            self.image_size
        )

        change_percentage = (
            changed_pixels /
            total_pixels
        ) * 100.0

        change_detected = (
            change_percentage > 1.0
        )

        print(
            f"BIT detection complete. "
            f"Changed pixels: {changed_pixels}, "
            f"Change: {change_percentage:.2f}%"
        )

        return {

            "change_detected":
                bool(change_detected),

            "change_percentage":
                round(change_percentage, 2),

            "changed_pixels":
                int(changed_pixels),

            "total_pixels":
                int(total_pixels),

            "image_size":
                [
                    self.image_size,
                    self.image_size
                ],

            "mask":
                full_mask,
        }