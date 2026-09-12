# app/services/change_analysis.py

import cv2
import numpy as np


class ChangeAnalyzer:

    def __init__(
        self,
        min_region_pixels=100
    ):

        self.min_region_pixels = (
            min_region_pixels
        )

    def analyze(self, change_mask):

        if change_mask is None:
            raise ValueError(
                "Change mask cannot be None."
            )

        mask = np.asarray(
            change_mask
        )

        # Handle BIT output if accidentally passed directly.
        if mask.ndim == 3:

            if mask.shape[0] == 2:
                mask = np.argmax(
                    mask,
                    axis=0
                )

            elif mask.shape[0] == 1:
                mask = mask[0]

            else:
                raise ValueError(
                    f"Unsupported mask shape: {mask.shape}"
                )

        if mask.ndim != 2:
            raise ValueError(
                f"Expected 2D mask, got {mask.shape}"
            )

        binary = (
            mask > 0
        ).astype(np.uint8)

        height, width = binary.shape

        total_pixels = height * width

        changed_pixels = int(
            binary.sum()
        )

        change_percentage = (
            changed_pixels /
            total_pixels
        ) * 100.0

        if changed_pixels == 0:

            return {
                "changed_pixels": 0,
                "total_pixels": total_pixels,
                "change_percentage": 0.0,
                "number_of_regions": 0,
                "largest_region_pixels": 0,
                "centroid": None,
                "bounding_box": None,
                "dominant_region": None,
                "severity": "none"
            }

        # --------------------------------------------------
        # Connected components
        # --------------------------------------------------

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        regions = []

        for label in range(
            1,
            num_labels
        ):

            area = int(
                stats[label, cv2.CC_STAT_AREA]
            )

            if area < self.min_region_pixels:
                continue

            x = int(
                stats[label, cv2.CC_STAT_LEFT]
            )

            y = int(
                stats[label, cv2.CC_STAT_TOP]
            )

            w = int(
                stats[label, cv2.CC_STAT_WIDTH]
            )

            h = int(
                stats[label, cv2.CC_STAT_HEIGHT]
            )

            cx, cy = centroids[label]

            regions.append({
                "area": area,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "centroid_x": float(cx),
                "centroid_y": float(cy)
            })

        regions.sort(
            key=lambda x: x["area"],
            reverse=True
        )

        largest_region = (
            regions[0]
            if regions
            else None
        )

        # --------------------------------------------------
        # Overall centroid
        # --------------------------------------------------

        ys, xs = np.where(
            binary > 0
        )

        centroid_x = float(
            np.mean(xs)
        )

        centroid_y = float(
            np.mean(ys)
        )

        # --------------------------------------------------
        # Overall bounding box
        # --------------------------------------------------

        x_min = int(xs.min())
        x_max = int(xs.max())

        y_min = int(ys.min())
        y_max = int(ys.max())

        # --------------------------------------------------
        # Region
        # --------------------------------------------------

        dominant_region = self._get_region(
            centroid_x,
            centroid_y,
            width,
            height
        )

        severity = self._get_severity(
            change_percentage
        )

        return {
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(
                change_percentage,
                2
            ),

            "number_of_regions": len(
                regions
            ),

            "largest_region_pixels": (
                largest_region["area"]
                if largest_region
                else 0
            ),

            "centroid": {
                "x": round(
                    centroid_x,
                    2
                ),
                "y": round(
                    centroid_y,
                    2
                )
            },

            "bounding_box": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            },

            "dominant_region": dominant_region,

            "severity": severity,

            "regions": regions[:20]
        }

    @staticmethod
    def _get_region(
        x,
        y,
        width,
        height
    ):

        horizontal = (
            "west"
            if x < width / 3
            else "east"
            if x > (2 * width / 3)
            else "central"
        )

        vertical = (
            "north"
            if y < height / 3
            else "south"
            if y > (2 * height / 3)
            else "central"
        )

        if vertical == "central" and horizontal == "central":
            return "central"

        return f"{vertical}-{horizontal}"

    @staticmethod
    def _get_severity(
        percentage
    ):

        if percentage == 0:
            return "none"

        if percentage < 1:
            return "minimal"

        if percentage < 5:
            return "low"

        if percentage < 15:
            return "moderate"

        if percentage < 30:
            return "high"

        return "extensive"