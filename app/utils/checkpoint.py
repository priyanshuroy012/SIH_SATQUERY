import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def get_bit_checkpoint():
    """
    Download the BIT checkpoint from Hugging Face if necessary.

    Returns:
        str: Local path to best_ckpt.pt
    """

    repo_id = os.getenv("BIT_HF_REPO_ID")

    if not repo_id:
        raise RuntimeError(
            "BIT_HF_REPO_ID environment variable is not configured."
        )

    filename = os.getenv(
        "BIT_HF_FILENAME",
        "best_ckpt.pt"
    )

    local_dir = Path(
        os.getenv(
            "BIT_CHECKPOINT_DIR",
            "models/checkpoints"
        )
    )

    local_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    local_path = local_dir / filename

    # -------------------------------------------------
    # Already downloaded
    # -------------------------------------------------

    if local_path.exists():
        print(
            f"BIT checkpoint already exists:\n"
            f"{local_path}"
        )

        return str(local_path)

    # -------------------------------------------------
    # Download from Hugging Face
    # -------------------------------------------------

    print(
        f"Downloading BIT checkpoint from:\n"
        f"{repo_id}"
    )

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir)
    )

    print(
        f"BIT checkpoint downloaded:\n"
        f"{downloaded_path}"
    )

    return str(downloaded_path)