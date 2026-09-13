from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    UploadFile,
    File
)

from app.services.explanation_service import (
    ExplanationService
)


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

    explanation_service = getattr(
        request.app.state,
        "explanation_service",
        None
    )

    startup_error = getattr(
        request.app.state,
        "startup_error",
        None
    )

    all_services_ready = (
        colab_bit_service is not None
        and explanation_service is not None
        and startup_error is None
    )

    return {

        "status": (
            "healthy"
            if all_services_ready
            else "unhealthy"
        ),

        "service": "SATQuery Model 2",

        "bit_backend": "Google Colab",

        "services": {

            "colab_bit": (
                "connected"
                if colab_bit_service is not None
                else "not_connected"
            ),

            "explanation": (
                "initialized"
                if explanation_service is not None
                else "not_initialized"
            )
        },

        "startup_error": startup_error
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

    # -----------------------------------------------------
    # Get Colab BIT service
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Validate services
    # -----------------------------------------------------

    if colab_bit_service is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Colab BIT service is not initialized."
            )
        )


    if explanation_service is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Explanation service is not initialized."
            )
        )


    # -----------------------------------------------------
    # Validate file types
    # -----------------------------------------------------

    allowed_types = {

        "image/png",

        "image/jpeg",

        "image/jpg"
    }


    if before_image.content_type not in allowed_types:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid before image format. "
                "Use PNG or JPEG."
            )
        )


    if after_image.content_type not in allowed_types:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid after image format. "
                "Use PNG or JPEG."
            )
        )


    before_path = None
    after_path = None


    try:

        # =================================================
        # 1. Save uploaded images temporarily
        # =================================================

        import tempfile
        import os


        before_suffix = (
            ".png"
            if before_image.content_type == "image/png"
            else ".jpg"
        )

        after_suffix = (
            ".png"
            if after_image.content_type == "image/png"
            else ".jpg"
        )


        before_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=before_suffix
        )

        after_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=after_suffix
        )


        before_path = before_temp.name
        after_path = after_temp.name


        # -------------------------------------------------
        # Write before image
        # -------------------------------------------------

        before_content = await before_image.read()

        before_temp.write(
            before_content
        )

        before_temp.close()


        # -------------------------------------------------
        # Write after image
        # -------------------------------------------------

        after_content = await after_image.read()

        after_temp.write(
            after_content
        )

        after_temp.close()


        # =================================================
        # 2. Send images to Colab BIT API
        # =================================================

        print(
            "Sending image pair to Colab BIT API..."
        )


        prediction = (
            colab_bit_service.detect(

                before_path=before_path,

                after_path=after_path
            )
        )


        # =================================================
        # 3. Extract Colab response
        # =================================================

        if not prediction:

            raise HTTPException(

                status_code=502,

                detail=(
                    "Colab BIT API returned "
                    "an empty response."
                )
            )


        if prediction.get("status") != "success":

            raise HTTPException(

                status_code=502,

                detail=(
                    "Colab BIT API failed: "
                    f"{prediction}"
                )
            )


        change_detection = (
            prediction.get(
                "change_detection",
                {}
            )
        )


        spatial_analysis = (
            prediction.get(
                "spatial_analysis",
                {}
            )
        )


        visualizations = (
            prediction.get(
                "visualizations",
                {}
            )
        )


        # =================================================
        # 4. Generate explanation
        # =================================================

        print(
            "Generating change explanation..."
        )


        explanation = (
            explanation_service.generate(

                analysis=spatial_analysis
            )
        )


        # =================================================
        # 5. Build final response
        # =================================================

        return {

            "status": "success",

            "input": {

                "before_filename": (
                    before_image.filename
                ),

                "after_filename": (
                    after_image.filename
                )
            },


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
                        0
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
                        "image_size",
                        []
                    )
                )
            },


            "spatial_analysis": {

                "change_percentage": (
                    spatial_analysis.get(
                        "change_percentage",
                        0
                    )
                ),

                "severity": (
                    spatial_analysis.get(
                        "severity",
                        "unknown"
                    )
                ),

                "dominant_region": (
                    spatial_analysis.get(
                        "dominant_region"
                    )
                ),

                "number_of_regions": (
                    spatial_analysis.get(
                        "number_of_regions",
                        0
                    )
                ),

                "largest_region_pixels": (
                    spatial_analysis.get(
                        "largest_region_pixels",
                        0
                    )
                )
            },


            "visualizations": {

                "mask": (
                    visualizations.get(
                        "mask"
                    )
                ),

                "overlay": (
                    visualizations.get(
                        "overlay"
                    )
                )
            },


            "explanation": explanation
        }


    except HTTPException:

        raise


    except Exception as e:

        print(
            "ERROR during change detection:"
        )

        print(
            repr(e)
        )


        raise HTTPException(

            status_code=500,

            detail=str(e)
        )


    finally:

        # =================================================
        # Cleanup temporary files
        # =================================================

        try:

            if before_path and os.path.exists(
                before_path
            ):

                os.remove(
                    before_path
                )


            if after_path and os.path.exists(
                after_path
            ):

                os.remove(
                    after_path
                )

        except Exception as cleanup_error:

            print(
                "Temporary file cleanup failed:"
            )

            print(
                repr(cleanup_error)
            )