from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    UploadFile,
    File
)

import os
import tempfile


router = APIRouter()


# =========================================================
# Health
# =========================================================

@router.get("/health")
def health(request: Request):

    colab_bit_service = getattr(
        request.app.state,
        "colab_bit_service",
        None
    )

    change_analyzer = getattr(
        request.app.state,
        "change_analyzer",
        None
    )

    explanation_service = getattr(
        request.app.state,
        "explanation_service",
        None
    )

    return {
        "status": "healthy",
        "service": "SATQuery Model 2",
        "input_mode": "image_upload",

        "services": {

            "colab_bit": (
                "initialized"
                if colab_bit_service is not None
                else "not_initialized"
            ),

            "change_analyzer": (
                "initialized"
                if change_analyzer is not None
                else "not_initialized"
            ),

            "explanation": (
                "initialized"
                if explanation_service is not None
                else "not_initialized"
            )
        }
    }


# =========================================================
# Detect Change
# =========================================================

@router.post("/detect-change")
async def detect_change(

    request: Request,

    before_image: UploadFile = File(...),

    after_image: UploadFile = File(...)

):

    # =====================================================
    # Get services initialized by main.py
    # =====================================================

    colab_bit_service = getattr(
        request.app.state,
        "colab_bit_service",
        None
    )

    explanation_service = getattr(
        request.app.state,
        "explanation_service",
        None
    )

    # =====================================================
    # Validate services
    # =====================================================

    if colab_bit_service is None:

        raise HTTPException(
            status_code=503,
            detail="Colab BIT service is unavailable."
        )

    if explanation_service is None:

        raise HTTPException(
            status_code=500,
            detail="Explanation service not initialized."
        )

    # =====================================================
    # Validate uploaded files
    # =====================================================

    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/jpg"
    }

    if before_image.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "before_image must be a PNG or JPEG image."
            )
        )

    if after_image.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "after_image must be a PNG or JPEG image."
            )
        )

    before_path = None
    after_path = None

    try:

        # =================================================
        # 1. Log request
        # =================================================

        print("\n" + "=" * 60)

        print(
            "SATQUERY MODEL 2 - IMAGE CHANGE DETECTION"
        )

        print("=" * 60)

        print(
            "Before image:",
            before_image.filename
        )

        print(
            "After image :",
            after_image.filename
        )

        # =================================================
        # 2. Read uploaded images
        # =================================================

        before_data = await before_image.read()

        after_data = await after_image.read()

        if not before_data:

            raise HTTPException(
                status_code=400,
                detail="before_image is empty."
            )

        if not after_data:

            raise HTTPException(
                status_code=400,
                detail="after_image is empty."
            )

        # =================================================
        # 3. Save temporary files
        # =================================================

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as before_file:

            before_file.write(before_data)

            before_path = before_file.name

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as after_file:

            after_file.write(after_data)

            after_path = after_file.name

        print(
            "Temporary before image:",
            before_path
        )

        print(
            "Temporary after image:",
            after_path
        )

        # =================================================
        # 4. Send images to Colab BIT
        # =================================================

        print(
            "\nRunning BIT change detection on Colab..."
        )

        prediction = colab_bit_service.detect(

            before_path=before_path,

            after_path=after_path
        )

        print(
            "BIT prediction received."
        )

        print(
            "BIT response keys:",
            prediction.keys()
        )

        # =================================================
        # 5. Extract change detection results
        # =================================================

        change_detection = prediction.get(
            "change_detection",
            {}
        )

        # =================================================
        # 6. Extract spatial analysis
        #
        # IMPORTANT:
        # Spatial analysis is now calculated inside Colab.
        # We no longer create placeholder values here.
        # =================================================

        spatial_analysis = prediction.get(
            "spatial_analysis"
        )

        if spatial_analysis is None:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Colab BIT service did not return "
                    "'spatial_analysis'. "
                    "Make sure the updated Colab endpoint "
                    "is running."
                )
            )

        print(
            "\nSpatial analysis received:"
        )

        print(
            spatial_analysis
        )

        # =================================================
        # 7. Validate spatial analysis
        # =================================================

        required_analysis_keys = [
            "change_percentage",
            "severity",
            "dominant_region",
            "number_of_regions",
            "largest_region_pixels"
        ]

        missing_keys = [
            key
            for key in required_analysis_keys
            if key not in spatial_analysis
        ]

        if missing_keys:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Colab spatial analysis is incomplete. "
                    f"Missing keys: {missing_keys}"
                )
            )

        # =================================================
        # 8. Generate explanation
        # =================================================

        print(
            "\nGenerating explanation..."
        )

        explanation = explanation_service.generate(

            analysis=spatial_analysis
        )

        print(
            "Explanation generated."
        )

        # =================================================
        # 9. Get visualizations
        # =================================================

        visualizations = prediction.get(
            "visualizations",
            {}
        )

        mask_base64 = visualizations.get(
            "mask"
        )

        overlay_base64 = visualizations.get(
            "overlay"
        )

        # =================================================
        # 10. Warn if visualizations are missing
        # =================================================

        if not mask_base64:

            print(
                "WARNING: Colab did not return "
                "a Base64 mask."
            )

        else:

            print(
                "Mask received."
            )

            print(
                "Mask Base64 length:",
                len(mask_base64)
            )

        if not overlay_base64:

            print(
                "WARNING: Colab did not return "
                "a Base64 overlay."
            )

        else:

            print(
                "Overlay received."
            )

            print(
                "Overlay Base64 length:",
                len(overlay_base64)
            )

        # =================================================
        # 11. Build final response
        # =================================================

        response = {

            "status": "success",

            "input": {

                "before_filename": (
                    before_image.filename
                ),

                "after_filename": (
                    after_image.filename
                )
            },

            # -------------------------------------------------
            # BIT change detection
            # -------------------------------------------------

            "change_detection": {

                "change_detected": (
                    change_detection.get(
                        "change_detected",
                        False
                    )
                ),

                "change_percentage": (
                    change_detection.get(
                        "change_percentage",
                        spatial_analysis.get(
                            "change_percentage",
                            0
                        )
                    )
                ),

                "changed_pixels": (
                    change_detection.get(
                        "changed_pixels",
                        0
                    )
                ),

                "total_pixels": (
                    change_detection.get(
                        "total_pixels",
                        0
                    )
                ),

                "image_size": (
                    change_detection.get(
                        "image_size"
                    )
                )
            },

            # -------------------------------------------------
            # Spatial analysis calculated by Colab
            # -------------------------------------------------

            "spatial_analysis": {

                "change_percentage": (
                    spatial_analysis[
                        "change_percentage"
                    ]
                ),

                "severity": (
                    spatial_analysis[
                        "severity"
                    ]
                ),

                "dominant_region": (
                    spatial_analysis[
                        "dominant_region"
                    ]
                ),

                "number_of_regions": (
                    spatial_analysis[
                        "number_of_regions"
                    ]
                ),

                "largest_region_pixels": (
                    spatial_analysis[
                        "largest_region_pixels"
                    ]
                )
            },

            # -------------------------------------------------
            # Base64 visualizations
            # -------------------------------------------------

            "visualizations": {

                "mask": mask_base64,

                "overlay": overlay_base64
            },

            # -------------------------------------------------
            # Human-readable explanation
            # -------------------------------------------------

            "explanation": explanation
        }

        # =================================================
        # 12. Finished
        # =================================================

        print(
            "\nChange detection completed successfully."
        )

        print("=" * 60 + "\n")

        return response

    # =====================================================
    # HTTP exceptions
    # =====================================================

    except HTTPException:

        raise

    # =====================================================
    # Unexpected errors
    # =====================================================

    except Exception as e:

        print(
            "\nERROR during change detection:"
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Change detection failed: "
                f"{type(e).__name__}: {str(e)}"
            )
        )

    # =====================================================
    # Cleanup
    # =====================================================

    finally:

        if (
            before_path is not None
            and os.path.exists(before_path)
        ):

            os.remove(before_path)

        if (
            after_path is not None
            and os.path.exists(after_path)
        ):

            os.remove(after_path)