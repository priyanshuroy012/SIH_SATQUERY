import os
import requests


class ColabBITService:

    def __init__(self):
        self.base_url = os.getenv("COLAB_BIT_URL")

        if not self.base_url:
            raise RuntimeError(
                "COLAB_BIT_URL is not configured."
            )

        self.base_url = self.base_url.rstrip("/")

        self.endpoint = (
            f"{self.base_url}/detect-change"
        )

    def health_check(self):
        response = requests.get(
            f"{self.base_url}/health",
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    def detect(
        self,
        before_path: str,
        after_path: str
    ):

        with open(before_path, "rb") as before_file, \
             open(after_path, "rb") as after_file:

            files = {
                "before_image": (
                    "before.png",
                    before_file,
                    "image/png"
                ),
                "after_image": (
                    "after.png",
                    after_file,
                    "image/png"
                )
            }

            response = requests.post(
                self.endpoint,
                files=files,
                timeout=180
            )

        response.raise_for_status()

        return response.json()