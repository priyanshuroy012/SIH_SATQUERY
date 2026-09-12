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

    # -----------------------------------------------------
    # Get services initialized by main.py
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Validate services
    # -----------------------------------------------------

    if colab_bit_service is None:

        raise HTTPException(
            status_code=503,
            detail="Colab BIT service is unavailable."
        )


    if change_analyzer is None:

        raise HTTPException(
            status_code=500,
            detail="Change analyzer not initialized."
        )


    if explanation_service is None:

        raise HTTPException(
            status_code=500,
            detail="Explanation service not initialized."
        )


    # -----------------------------------------------------
    # Validate uploaded files
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
        # 1. Read uploaded images
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
        # 2. Save temporary files
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
        # 3. Run BIT on Colab GPU
        # =================================================

        print(
            "\nRunning BIT change detection..."
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
        # 4. Extract BIT change detection results
        # =================================================

        change_detection = prediction.get(
            "change_detection",
            {}
        )


        # -------------------------------------------------
        # Check whether Colab returned a mask
        # -------------------------------------------------

        change_mask = prediction.get(
            "change_mask"
        )


        # =================================================
        # 5. Spatial analysis
        # =================================================

        print(
            "\nRunning spatial analysis..."
        )


        if change_mask is not None:

            analysis = change_analyzer.analyze(
                change_mask
            )

        else:

            # ------------------------------------------------
            # Colab currently returns statistics + visualizations
            # rather than the numerical mask itself.
            #
            # Build the minimum analysis dictionary required
            # by ExplanationService.
            # ------------------------------------------------

            change_percentage = change_detection.get(
                "change_percentage",
                0
            )


            # Determine severity

            if change_percentage == 0:

                severity = "none"

            elif change_percentage < 2:

                severity = "low"

            elif change_percentage < 10:

                severity = "moderate"

            elif change_percentage < 25:

                severity = "high"

            else:

                severity = "very high"


            analysis = {

                "change_percentage": (
                    change_percentage
                ),

                "severity": severity,

                "dominant_region": None,

                "number_of_regions": 0,

                "largest_region_pixels": 0

            }


        print(
            "Spatial analysis:",
            analysis
        )


        # =================================================
        # 6. Generate explanation
        # =================================================

        print(
            "\nGenerating explanation..."
        )


        explanation = explanation_service.generate(

            analysis=analysis

        )


        print(
            "Explanation generated."
        )


        # =================================================
        # 7. Get visualizations
        # =================================================

        visualizations = prediction.get(
            "visualizations",
            {}
        )


        # =================================================
        # 8. Build final response
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
                        "image_size"
                    )
                )

            },


            "spatial_analysis": analysis,


            "visualizations": {

                "mask": visualizations.get(
                    "mask"
                ),

                "overlay": visualizations.get(
                    "overlay"
                )

            },


            "explanation": explanation

        }


        print(
            "\nChange detection completed successfully."
        )

        print("=" * 60 + "\n")


        return response


    except HTTPException:

        raise


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


    finally:

        # =================================================
        # Cleanup temporary files
        # =================================================

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