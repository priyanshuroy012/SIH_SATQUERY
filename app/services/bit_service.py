# app/services/bit_service.py

import numpy as np

from app.models.bit_model import BITModel
from app.services.preprocessing import ImagePreprocessor


class BITService:

    """
    High-level service for running BIT change detection
    on full satellite images.

    Workflow:

        T1 image
            ↓
        resize
            ↓
        16 × 256x256 patches
            ↓
        BIT

        T2 image
            ↓
        resize
            ↓
        16 × 256x256 patches
            ↓
        BIT

            ↓
        reconstruct 1024x1024 mask
    """

    def __init__(
        self,
        checkpoint_path,
        bit_repo_path=None,
        patch_size=256,
        image_size=1024,
        device=None
    ):

        self.patch_size = patch_size

        self.image_size = image_size


        # -------------------------------------------------
        # Image preprocessing
        # -------------------------------------------------

        self.preprocessor = (
            ImagePreprocessor(

                patch_size=patch_size,

                image_size=image_size
            )
        )


        # -------------------------------------------------
        # Load BIT model
        # -------------------------------------------------

        print(
            "Creating BIT model..."
        )

        self.model = BITModel(

            checkpoint_path=checkpoint_path,

            bit_repo_path=bit_repo_path,

            device=device
        )


        print(
            "BITService initialized successfully."
        )


    # =====================================================
    # Main detection function
    # =====================================================

    def detect(
        self,
        before_path,
        after_path
    ):

        # -------------------------------------------------
        # 1. Load images
        # -------------------------------------------------

        before = (
            self.preprocessor.load_image(
                before_path
            )
        )

        after = (
            self.preprocessor.load_image(
                after_path
            )
        )


        # -------------------------------------------------
        # 2. Resize images
        # -------------------------------------------------

        before = (
            self.preprocessor.resize(
                before
            )
        )

        after = (
            self.preprocessor.resize(
                after
            )
        )


        # -------------------------------------------------
        # 3. Extract patches
        # -------------------------------------------------

        before_patches = (
            self.preprocessor.extract_patches(
                before
            )
        )

        after_patches = (
            self.preprocessor.extract_patches(
                after
            )
        )


        # -------------------------------------------------
        # 4. Verify patch counts
        # -------------------------------------------------

        if (
            len(before_patches)
            != len(after_patches)
        ):

            raise RuntimeError(
                "Before/after patch count mismatch."
            )


        # -------------------------------------------------
        # 5. Full reconstructed mask
        # -------------------------------------------------

        full_prediction = np.zeros(

            (
                self.image_size,
                self.image_size
            ),

            dtype=np.uint8
        )


        # -------------------------------------------------
        # 6. Run BIT on every patch
        # -------------------------------------------------

        for (
            before_patch,
            after_patch
        ) in zip(

            before_patches,
            after_patches
        ):

            # ---------------------------------------------
            # Add batch dimension
            # ---------------------------------------------

            before_tensor = (
                before_patch["tensor"]
                .unsqueeze(0)
            )

            after_tensor = (
                after_patch["tensor"]
                .unsqueeze(0)
            )


            # ---------------------------------------------
            # IMPORTANT:
            #
            # BITModel itself has predict_mask()
            #
            # NOT:
            # self.model.model.predict_mask()
            #
            # ---------------------------------------------

            prediction = (
                self.model.predict_mask(

                    before_tensor,

                    after_tensor
                )
            )


            # ---------------------------------------------
            # Tensor → NumPy
            # ---------------------------------------------

            prediction = (

                prediction

                .squeeze()

                .detach()

                .cpu()

                .numpy()

                .astype(np.uint8)
            )


            # ---------------------------------------------
            # Patch coordinates
            # ---------------------------------------------

            x = before_patch["x"]

            y = before_patch["y"]


            # ---------------------------------------------
            # Reconstruct full mask
            # ---------------------------------------------

            full_prediction[

                y:y + self.patch_size,

                x:x + self.patch_size

            ] = prediction


        # -------------------------------------------------
        # 7. Calculate statistics
        # -------------------------------------------------

        changed_pixels = int(

            np.sum(
                full_prediction > 0
            )
        )


        total_pixels = int(

            self.image_size
            * self.image_size
        )


        change_percentage = (

            changed_pixels
            / total_pixels
        ) * 100.0


        # -------------------------------------------------
        # 8. Determine whether meaningful change exists
        # -------------------------------------------------

        change_detected = (

            change_percentage > 1.0
        )


        # -------------------------------------------------
        # 9. Return result
        # -------------------------------------------------

        return {

            "change_detected": bool(
                change_detected
            ),

            "change_percentage": float(
                change_percentage
            ),

            "changed_pixels": int(
                changed_pixels
            ),

            "total_pixels": int(
                total_pixels
            ),

            "change_mask": (
                full_prediction
            ),

            "image_size": (

                self.image_size,

                self.image_size
            )
        }