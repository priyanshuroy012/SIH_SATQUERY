# app/services/explanation_service.py


class ExplanationService:

    def generate(
        self,
        analysis,
        before_date=None,
        after_date=None
    ):

        percentage = analysis[
            "change_percentage"
        ]

        severity = analysis[
            "severity"
        ]

        region = analysis[
            "dominant_region"
        ]

        number_of_regions = analysis[
            "number_of_regions"
        ]

        if percentage == 0:

            explanation = (
                "No significant pixel-level change "
                "was detected between the two observations."
            )

        else:

            region_text = (
                region
                if region
                else "the observed scene"
            )

            explanation = (
                f"Approximately {percentage:.2f}% "
                f"of the observed area shows detectable "
                f"pixel-level change. "
                f"The overall change severity is "
                f"{severity}. "
                f"The detected changes are primarily "
                f"concentrated in the {region_text} "
                f"portion of the scene. "
                f"The detected change consists of "
                f"{number_of_regions} spatially connected "
                f"region(s)."
            )

        if before_date and after_date:

            explanation += (
                f" The comparison uses observations "
                f"from {before_date} and {after_date}."
            )

        return explanation

    def generate_detailed(
        self,
        analysis,
        before_date=None,
        after_date=None
    ):

        percentage = analysis[
            "change_percentage"
        ]

        severity = analysis[
            "severity"
        ]

        regions = analysis[
            "number_of_regions"
        ]

        largest_region = analysis[
            "largest_region_pixels"
        ]

        return {
            "summary": self.generate(
                analysis,
                before_date,
                after_date
            ),

            "change_percentage": percentage,

            "severity": severity,

            "spatial_regions": regions,

            "largest_region_pixels": (
                largest_region
            ),

            "semantic_interpretation": (
                "BIT provides pixel-level change "
                "detection. It does not independently "
                "identify the semantic cause of the "
                "change, such as construction, demolition, "
                "vegetation loss, or road development."
            )
        }