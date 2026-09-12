import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


# =========================================================
# Locate BIT_CD repository
# =========================================================

# Expected structure:
#
# SIH/
# ├── BIT_CD/
# │   ├── models/
# │   │   ├── basic_model.py
# │   │   └── networks.py
# │   │
# │   └── checkpoints/
# │       └── BIT_LEVIR/
# │           └── best_ckpt.pt
# │
# └── app/
#     └── models/
#         └── bit_model.py


BIT_REPO_PATH = Path(__file__).resolve().parents[2] / "BIT_CD"


if not BIT_REPO_PATH.exists():
    raise FileNotFoundError(
        f"BIT_CD repository not found at:\n{BIT_REPO_PATH}"
    )


# Add BIT_CD to Python's module search path
if str(BIT_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(BIT_REPO_PATH))


# =========================================================
# Import original BIT implementation
# =========================================================

from models.basic_model import CDEvaluator


class BITModel:
    """
    Wrapper around the original BIT-CD implementation.

    Architecture:
        base_transformer_pos_s4_dd8_dedim8

    Number of classes:
        2

    Class 0:
        unchanged

    Class 1:
        changed
    """

    def __init__(
        self,
        checkpoint_path: str,
        bit_repo_path: str = None,
        device: str = None,
    ):

        # -------------------------------------------------
        # Optional custom BIT repository path
        # -------------------------------------------------

        if bit_repo_path:

            custom_repo = Path(bit_repo_path).resolve()

            if not custom_repo.exists():
                raise FileNotFoundError(
                    f"BIT repository not found at:\n{custom_repo}"
                )

            if str(custom_repo) not in sys.path:
                sys.path.insert(0, str(custom_repo))


        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        self.device = self._get_device(device)

        print(f"BIT device: {self.device}")


        # -------------------------------------------------
        # Checkpoint
        # -------------------------------------------------

        checkpoint_path = os.path.abspath(
            checkpoint_path
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"BIT checkpoint not found:\n{checkpoint_path}"
            )

        print(
            f"BIT checkpoint found:\n{checkpoint_path}"
        )


        # -------------------------------------------------
        # Checkpoint directory
        # -------------------------------------------------

        checkpoint_dir = os.path.dirname(
            checkpoint_path
        )

        checkpoint_name = os.path.basename(
            checkpoint_path
        )


        # -------------------------------------------------
        # GPU configuration
        # -------------------------------------------------

        gpu_ids = []

        if self.device.type == "cuda":
            gpu_ids = [
                self.device.index or 0
            ]


        # -------------------------------------------------
        # Arguments expected by original BIT code
        # -------------------------------------------------

        args = SimpleNamespace(

            n_class=2,

            net_G=(
                "base_transformer_pos_s4_dd8_dedim8"
            ),

            gpu_ids=gpu_ids,

            checkpoint_dir=checkpoint_dir,

            output_folder=(
                "outputs/bit_predictions"
            )
        )


        # -------------------------------------------------
        # Initialize original BIT evaluator
        # -------------------------------------------------

        print(
            "Initializing BIT CDEvaluator..."
        )

        self.evaluator = CDEvaluator(
            args=args
        )


        # -------------------------------------------------
        # Load checkpoint
        # -------------------------------------------------

        self._load_checkpoint(
            checkpoint_name=checkpoint_name
        )


        # -------------------------------------------------
        # Evaluation mode
        # -------------------------------------------------

        self.evaluator.eval()

        self.model = (
            self.evaluator.net_G
        )

        self.model.to(
            self.device
        )

        self.model.eval()


        print(
            "BIT model loaded successfully."
        )


    # =====================================================
    # Device selection
    # =====================================================

    @staticmethod
    def _get_device(device=None):

        # User explicitly requested a device
        if device:

            if (
                device.startswith("cuda")
                and torch.cuda.is_available()
            ):
                return torch.device(device)

            return torch.device("cpu")


        # Automatically use GPU if available
        if torch.cuda.is_available():

            return torch.device(
                "cuda:0"
            )


        # Otherwise CPU
        return torch.device("cpu")


    # =====================================================
    # Load checkpoint
    # =====================================================

    def _load_checkpoint(
        self,
        checkpoint_name
    ):

        checkpoint_path = os.path.join(

            self.evaluator.checkpoint_dir,

            checkpoint_name
        )

        print(
            f"Loading BIT checkpoint:\n"
            f"{checkpoint_path}"
        )


        checkpoint = torch.load(

            checkpoint_path,

            map_location=self.device,

            # Required for compatibility with
            # older BIT checkpoints on newer PyTorch
            weights_only=False
        )


        # -------------------------------------------------
        # Validate checkpoint
        # -------------------------------------------------

        if (
            "model_G_state_dict"
            not in checkpoint
        ):

            raise KeyError(
                "BIT checkpoint does not contain "
                "'model_G_state_dict'."
            )


        # -------------------------------------------------
        # Load model weights
        # -------------------------------------------------

        self.model = (
            self.evaluator.net_G
        )

        self.model.load_state_dict(
            checkpoint[
                "model_G_state_dict"
            ]
        )


        self.model.to(
            self.device
        )

        self.model.eval()


        print(
            "BIT checkpoint loaded successfully."
        )


    # =====================================================
    # Run BIT on one patch pair
    # =====================================================

    @torch.no_grad()
    def predict(
        self,
        before_tensor,
        after_tensor
    ):

        """
        Input:
            before_tensor:
                [1, 3, 256, 256]

            after_tensor:
                [1, 3, 256, 256]

        Output:
            logits:
                [1, 2, 256, 256]
        """

        before_tensor = (
            before_tensor.to(
                self.device
            )
        )

        after_tensor = (
            after_tensor.to(
                self.device
            )
        )


        logits = self.model(

            before_tensor,

            after_tensor
        )


        return logits


    # =====================================================
    # Convert logits to binary change mask
    # =====================================================

    @torch.no_grad()
    def predict_mask(
        self,
        before_tensor,
        after_tensor
    ):

        logits = self.predict(

            before_tensor,

            after_tensor
        )


        # Class with highest probability:
        #
        # 0 = unchanged
        # 1 = changed

        mask = torch.argmax(

            logits,

            dim=1
        )


        return mask


    # =====================================================
    # Device information
    # =====================================================

    def get_device(self):

        return str(
            self.device
        )