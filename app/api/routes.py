import os
import tempfile

from fastapi import APIRouter, File, UploadFile, Request, HTTPException

router = APIRouter()


@router.post("/detect-change")
async def detect_change(
    request: Request,
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...)
):

    # -----------------------------------------
    # Validate file types
    # -----------------------------------------

    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/jpg"
    }

    if before_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="before_image must be PNG or JPEG."
        )

    if after_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="after_image must be PNG or JPEG."
        )

    # -----------------------------------------
    # Get services
    # -----------------------------------------

    colab_bit_service = request.app.state.colab_bit_service
    change_analyzer = request.app.state.change_analyzer
    explanation_service = request.app.state.explanation_service

    if colab_bit_service is None:
        raise HTTPException(
            status_code=503,
            detail="Colab BIT service is unavailable."
        )

    # -----------------------------------------
    # Read uploaded images
    # -----------------------------------------

    before_data = await before_image.read()
    after_data = await after_image.read()

    before_path = tempfile.mktemp(
        suffix=".png"
    )

    after_path = tempfile.mktemp(
        suffix=".png"
    )

    try:

        # -----------------------------------------
        # Save temporary files
        # -----------------------------------------

        with open(before_path, "wb") as f:
            f.write(before_data)

        with open(after_path, "wb") as f:
            f.write(after_data)

        # -----------------------------------------
        # Send images to Colab BIT
        # -----------------------------------------

        prediction = colab_bit_service.detect(
            before_path,
            after_path
        )

        # -----------------------------------------
        # Extract change detection result
        # -----------------------------------------

        change_detection = prediction.get(
            "change_detection",
            {}
        )

        visualizations = prediction.get(
            "visualizations",
            {}
        )

        # -----------------------------------------
        # Statistics
        # -----------------------------------------

        change_percentage = float(
            change_detection.get(
                "change_percentage",
                0
            )
        )

        changed_pixels = int(
            change_detection.get(
                "changed_pixels",
                0
            )
        )

        total_pixels = int(
            change_detection.get(
                "total_pixels",
                0
            )
        )

        change_detected = bool(
            change_detection.get(
                "change_detected",
                False
            )
        )

        # -----------------------------------------
        # Severity
        # -----------------------------------------

        if change_percentage == 0:
            severity = "none"

        elif change_percentage < 1:
            severity = "low"

        elif change_percentage < 5:
            severity = "moderate"

        elif change_percentage < 15:
            severity = "high"

        else:
            severity = "very_high"

        # -----------------------------------------
        # Spatial analysis
        # -----------------------------------------

        spatial_analysis = {
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": change_percentage,
            "change_detected": change_detected,

            "number_of_regions": None,
            "largest_region_pixels": None,

            "centroid": None,
            "bounding_box": None,
            "dominant_region": None,

            "severity": severity,

            "image_size": change_detection.get(
                "image_size",
                [1024, 1024]
            )
        }

        # -----------------------------------------
        # Explanation
        # -----------------------------------------

        explanation = explanation_service.generate(
            change_detection=change_detection,
            spatial_analysis=spatial_analysis
        )

        # -----------------------------------------
        # Final response
        # -----------------------------------------

        return {
            "status": "success",

            "input": {
                "before_filename": before_image.filename,
                "after_filename": after_image.filename
            },

            "change_detection": change_detection,

            "spatial_analysis": spatial_analysis,

            "visualizations": {
                "mask": visualizations.get("mask"),
                "overlay": visualizations.get("overlay")
            },

            "explanation": explanation
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Change detection failed: {str(e)}"
        )

    finally:

        if os.path.exists(before_path):
            os.remove(before_path)

        if os.path.exists(after_path):
            os.remove(after_path)