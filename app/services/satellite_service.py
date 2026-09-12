# app/services/satellite_service.py

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
    MosaickingOrder,
)


# =========================================================
# Load environment variables
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


class SatelliteService:

    def __init__(
        self,
        output_dir="data/downloads",
        bbox_size_meters=1024,
        resolution=10,
    ):

        # =================================================
        # Output configuration
        # =================================================

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.bbox_size_meters = bbox_size_meters
        self.resolution = resolution

        # =================================================
        # Read credentials
        # =================================================

        client_id = os.getenv("SH_CLIENT_ID")
        client_secret = os.getenv("SH_CLIENT_SECRET")

        base_url = os.getenv(
            "SH_BASE_URL",
            "https://sh.dataspace.copernicus.eu",
        )

        token_url = os.getenv(
            "SH_TOKEN_URL",
            "https://identity.dataspace.copernicus.eu/"
            "auth/realms/CDSE/protocol/openid-connect/token",
        )

        # =================================================
        # Validate credentials
        # =================================================

        if not client_id:
            raise RuntimeError(
                "SH_CLIENT_ID is not configured."
            )

        if not client_secret:
            raise RuntimeError(
                "SH_CLIENT_SECRET is not configured."
            )

        # =================================================
        # Configure Sentinel Hub
        # =================================================

        self.config = SHConfig()

        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret

        self.config.sh_base_url = base_url
        self.config.sh_token_url = token_url

        # =================================================
        # Print configuration for debugging
        # =================================================

        print("=" * 60)
        print("Sentinel Hub configuration")
        print("=" * 60)

        print(
            f"Client ID configured: "
            f"{bool(self.config.sh_client_id)}"
        )

        print(
            f"Client secret configured: "
            f"{bool(self.config.sh_client_secret)}"
        )

        print(
            f"SH base URL: "
            f"{self.config.sh_base_url}"
        )

        print(
            f"SH token URL: "
            f"{self.config.sh_token_url}"
        )

        print("=" * 60)

    # =====================================================
    # Create bounding box
    # =====================================================

    def _create_bbox(
        self,
        latitude,
        longitude,
    ):

        # Approximate conversion:
        # 1 degree latitude ≈ 111 km

        lat_delta = (
            self.bbox_size_meters
            / 2
            / 111_000
        )

        lon_delta = (
            self.bbox_size_meters
            / 2
            / (
                111_000
                * np.cos(
                    np.radians(latitude)
                )
            )
        )

        bbox = BBox(
            bbox=[
                longitude - lon_delta,
                latitude - lat_delta,
                longitude + lon_delta,
                latitude + lat_delta,
            ],
            crs=CRS.WGS84,
        )

        return bbox

    # =====================================================
    # Sentinel Hub evalscript
    # =====================================================

    def _evalscript(self):

        return """
        //VERSION=3

        function setup() {

            return {

                input: [{
                    bands: [
                        "B02",
                        "B03",
                        "B04",
                        "SCL"
                    ]
                }],

                output: {
                    bands: 3,
                    sampleType: "UINT8"
                }
            };
        }

        function evaluatePixel(sample) {

            return [
                255 * sample.B04,
                255 * sample.B03,
                255 * sample.B02
            ];
        }
        """

    # =====================================================
    # Request one image
    # =====================================================

    def _request_image(
        self,
        bbox,
        target_date,
        max_cloud_cover,
    ):

        start_date = (
            target_date
            - timedelta(days=30)
        )

        end_date = (
            target_date
            + timedelta(days=30)
        )

        size = bbox_to_dimensions(
            bbox,
            resolution=self.resolution,
        )

        print(
            f"Requesting Sentinel-2 image "
            f"around {target_date}"
        )

        print(
            f"Date window: "
            f"{start_date} → {end_date}"
        )

        print(
            f"Output size: {size}"
        )

        print(
            f"Cloud limit: "
            f"{max_cloud_cover}%"
        )

        print(
            f"Request endpoint: "
            f"{self.config.sh_base_url}/api/v1/process"
        )

        # =================================================
        # IMPORTANT:
        #
        # Explicitly bind Sentinel-2 collection to the
        # CDSE Sentinel Hub service.
        # =================================================

        sentinel2_collection = (
            DataCollection.SENTINEL2_L2A.define_from(
                "s2l2a",
                service_url=self.config.sh_base_url,
            )
        )

        request = SentinelHubRequest(

            evalscript=self._evalscript(),

            input_data=[
                SentinelHubRequest.input_data(

                    data_collection=(
                        sentinel2_collection
                    ),

                    time_interval=(
                        start_date.isoformat(),
                        end_date.isoformat(),
                    ),

                    mosaicking_order=(
                        MosaickingOrder.LEAST_CC
                    ),

                    other_args={
                        "dataFilter": {
                            "maxCloudCoverage": (
                                max_cloud_cover
                            )
                        }
                    },
                )
            ],

            responses=[
                SentinelHubRequest.output_response(
                    "default",
                    MimeType.PNG,
                )
            ],

            bbox=bbox,

            size=size,

            config=self.config,
        )

        # =================================================
        # Download
        # =================================================

        images = request.get_data()

        if not images:

            raise RuntimeError(
                f"No Sentinel-2 image found "
                f"around {target_date}"
            )

        return images[0]

    # =====================================================
    # Public API
    # =====================================================

    def get_images(
        self,
        latitude,
        longitude,
        before_date,
        after_date,
        max_cloud_cover=20,
    ):

        # =================================================
        # Convert strings to date objects
        # =================================================

        if isinstance(
            before_date,
            str,
        ):

            before_date = date.fromisoformat(
                before_date
            )

        if isinstance(
            after_date,
            str,
        ):

            after_date = date.fromisoformat(
                after_date
            )

        # =================================================
        # Validate dates
        # =================================================

        if before_date >= after_date:

            raise ValueError(
                "before_date must be earlier "
                "than after_date."
            )

        # =================================================
        # Create geographic area
        # =================================================

        bbox = self._create_bbox(
            latitude,
            longitude,
        )

        # =================================================
        # Download before image
        # =================================================

        before_image = self._request_image(
            bbox=bbox,
            target_date=before_date,
            max_cloud_cover=max_cloud_cover,
        )

        # =================================================
        # Download after image
        # =================================================

        after_image = self._request_image(
            bbox=bbox,
            target_date=after_date,
            max_cloud_cover=max_cloud_cover,
        )

        # =================================================
        # File names
        # =================================================

        location_id = (
            f"{latitude:.5f}_"
            f"{longitude:.5f}"
        )

        before_path = (
            self.output_dir
            / f"{location_id}_before.png"
        )

        after_path = (
            self.output_dir
            / f"{location_id}_after.png"
        )

        # =================================================
        # Save images
        # =================================================

        Image.fromarray(
            before_image
        ).save(
            before_path
        )

        Image.fromarray(
            after_image
        ).save(
            after_path
        )

        # =================================================
        # Return information
        # =================================================

        return {

            "before_image": str(
                before_path
            ),

            "after_image": str(
                after_path
            ),

            "requested_before_date": (
                before_date.isoformat()
            ),

            "requested_after_date": (
                after_date.isoformat()
            ),

            "bbox": list(
                bbox
            ),
        }