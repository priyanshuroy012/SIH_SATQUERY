from fastapi import APIRouter, Request, UploadFile, File, HTTPException
import os
import tempfile

router = APIRouter()


@router.get("/health")
def health(request: Request):
    try:
        colab_service = request.app.state.colab_bit_service
        colab_health = colab_service.health_check()

        return {
            "status": "ok",
            "colab_bit": colab_health,
            "explanation": "available"
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service health check failed: {str(e)}"
        )


@router.post("/detect-change")
async def detect_change(
    request: Request,
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...)
):

    # --------------------------------------------------
    # Validate file types
    # --------------------------------------------------

    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/jpg"
    }

    if before_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="before_image must be PNG or JPEG"
        )

    if after_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="after_image must be PNG or JPEG"
        )

    before_path = None
    after_path = None

    try:

        # --------------------------------------------------
        # Save uploaded images temporarily
        # --------------------------------------------------

        before_suffix = ".png" if before_image.content_type == "image/png" else ".jpg"
        after_suffix = ".png" if after_image.content_type == "image/png" else ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=before_suffix
        ) as before_temp:

            before_temp.write(await before_image.read())
            before_path = before_temp.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=after_suffix
        ) as after_temp:

            after_temp.write(await after_image.read())
            after_path = after_temp.name

        # --------------------------------------------------
        # Get services
        # --------------------------------------------------

        colab_service = request.app.state.colab_bit_service
        explanation_service = request.app.state.explanation_service

        # --------------------------------------------------
        # Send images to Colab BIT
        # --------------------------------------------------

        colab_result = colab_service.detect(
            before_path,
            after_path
        )

        print("========== COLAB RESPONSE ==========")
        print(colab_result)
        print("====================================")

        # --------------------------------------------------
        # Extract response sections
        # --------------------------------------------------

        change_detection = colab_result.get(
            "change_detection",
            {}
        )

        spatial_analysis = colab_result.get(
            "spatial_analysis",
            {}
        )

        visualizations = colab_result.get(
            "visualizations",
            {}
        )

        # --------------------------------------------------
        # Safety fallback
        #
        # If Colab did not return spatial_analysis,
        # construct it from change_detection.
        # --------------------------------------------------

        if not spatial_analysis:

            spatial_analysis = {
                "change_percentage": change_detection.get(
                    "change_percentage",
                    0
                ),
                "severity": "unknown",
                "dominant_region": None,
                "number_of_regions": 0,
                "largest_region_pixels": 0
            }

        # --------------------------------------------------
        # Make sure required fields exist
        # --------------------------------------------------

        spatial_analysis.setdefault(
            "change_percentage",
            change_detection.get("change_percentage", 0)
        )

        spatial_analysis.setdefault(
            "severity",
            "unknown"
        )

        spatial_analysis.setdefault(
            "dominant_region",
            None
        )

        spatial_analysis.setdefault(
            "number_of_regions",
            0
        )

        spatial_analysis.setdefault(
            "largest_region_pixels",
            0
        )

        # --------------------------------------------------
        # Generate explanation
        # --------------------------------------------------

        explanation = explanation_service.generate(
            analysis=spatial_analysis
        )

        # --------------------------------------------------
        # Final response
        # --------------------------------------------------

        return {
            "status": "success",

            "change_detection": change_detection,

            "spatial_analysis": spatial_analysis,

            "visualizations": visualizations,

            "explanation": explanation
        }

    except Exception as e:

        print("========== DETECTION ERROR ==========")
        print(repr(e))
        print("=====================================")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # --------------------------------------------------
        # Cleanup temporary files
        # --------------------------------------------------

        if before_path and os.path.exists(before_path):
            os.remove(before_path)

        if after_path and os.path.exists(after_path):
            os.remove(after_path)