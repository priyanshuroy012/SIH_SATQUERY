from fastapi import (
    APIRouter,
    HTTPException,
    Request
)

from app.schemas.request_response import (
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    Location,
    ImageDates
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

    satellite_service = getattr(
        request.app.state,
        "satellite_service",
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

    startup_error = getattr(
        request.app.state,
        "startup_error",
        None
    )

    all_services_ready = all([
        satellite_service is not None,
        colab_bit_service is not None,
        change_analyzer is not None,
        explanation_service is not None
    ])

    status = (
        "healthy"
        if all_services_ready and startup_error is None
        else "unhealthy"
    )

    return {

        "status": status,

        "service": "SATQuery Model 2",

        "inference": "Google Colab",

        "services": {

            "satellite": (
                "initialized"
                if satellite_service is not None
                else "not_initialized"
            ),

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
        },

        "startup_error": startup_error
    }


# =========================================================
# Detect change
# =========================================================

@router.post(
    "/detect-change",
    response_model=ChangeDetectionResponse
)
def detect_change(

    request: Request,

    data: ChangeDetectionRequest

):

    # -----------------------------------------------------
    # Get services initialized by main.py
    # -----------------------------------------------------

    satellite_service = getattr(
        request.app.state,
        "satellite_service",
        None
    )

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

    if satellite_service is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Satellite service not initialized."
            )
        )


    if colab_bit_service is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Colab BIT service not initialized."
            )
        )


    if change_analyzer is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Change analyzer not initialized."
            )
        )


    if explanation_service is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Explanation service not initialized."
            )
        )


    try:

        # =================================================
        # 1. Retrieve satellite imagery
        # =================================================

        print(
            "Retrieving satellite imagery..."
        )

        satellite_result = (
            satellite_service.get_images(

                latitude=data.latitude,

                longitude=data.longitude,

                before_date=data.before_date,

                after_date=data.after_date,

                max_cloud_cover=(
                    data.max_cloud_cover
                )
            )
        )


        # =================================================
        # 2. Send images to Colab BIT API
        # =================================================

        print(
            "Sending satellite images to "
            "Colab BIT inference..."
        )

        prediction = (
            colab_bit_service.detect(

                before_path=(
                    satellite_result[
                        "before_image"
                    ]
                ),

                after_path=(
                    satellite_result[
                        "after_image"
                    ]
                )
            )
        )


        print(
            "Colab BIT inference completed."
        )

        print(
            f"BIT prediction: {prediction}"
        )


        # =================================================
        # 3. Build spatial analysis
        # =================================================
        #
        # Current Colab endpoint returns:
        #
        # - change_detected
        # - change_percentage
        # - changed_pixels
        # - total_pixels
        # - image_size
        #
        # It does NOT currently return the complete
        # change_mask.
        #
        # Therefore we create a statistics-based analysis
        # object here.
        #
        # =================================================

        change_detection = prediction.get(
            "change_detection",
            {}
        )

        changed_pixels = change_detection.get(
            "changed_pixels",
            0
        )

        total_pixels = change_detection.get(
            "total_pixels",
            0
        )

        change_percentage = change_detection.get(
            "change_percentage",
            0
        )

        change_detected = change_detection.get(
            "change_detected",
            False
        )

        image_size = change_detection.get(
            "image_size",
            [1024, 1024]
        )


        # -------------------------------------------------
        # Determine simple severity
        # -------------------------------------------------

        if change_percentage <= 0:

            severity = "none"

        elif change_percentage < 1:

            severity = "low"

        elif change_percentage < 5:

            severity = "moderate"

        elif change_percentage < 15:

            severity = "high"

        else:

            severity = "very_high"


        analysis = {

            "changed_pixels": (
                changed_pixels
            ),

            "total_pixels": (
                total_pixels
            ),

            "change_percentage": (
                change_percentage
            ),

            "change_detected": (
                change_detected
            ),

            "number_of_regions": None,

            "largest_region_pixels": None,

            "centroid": None,

            "bounding_box": None,

            "dominant_region": None,

            "severity": severity,

            "image_size": image_size,

            "note": (
                "Spatial region analysis is not "
                "available because the Colab inference "
                "API currently returns change statistics "
                "rather than the full change mask."
            )
        }


        # =================================================
        # 4. Generate explanation
        # =================================================

        print(
            "Generating explanation..."
        )

        explanation = (
            explanation_service.generate(

                analysis=analysis,

                before_date=(
                    data.before_date.isoformat()
                ),

                after_date=(
                    data.after_date.isoformat()
                )
            )
        )


        # =================================================
        # 5. Build final API response
        # =================================================

        return ChangeDetectionResponse(

            status="success",

            location=Location(

                latitude=data.latitude,

                longitude=data.longitude
            ),

            dates=ImageDates(

                requested_before=(
                    data.before_date.isoformat()
                ),

                requested_after=(
                    data.after_date.isoformat()
                ),

                actual_before=(
                    satellite_result.get(
                        "actual_before_date"
                    )
                ),

                actual_after=(
                    satellite_result.get(
                        "actual_after_date"
                    )
                )
            ),

            change_detection={

                "change_detected": (
                    change_detected
                ),

                "change_percentage": (
                    change_percentage
                ),

                "changed_pixels": (
                    changed_pixels
                ),

                "total_pixels": (
                    total_pixels
                ),

                "image_size": (
                    image_size
                )
            },

            spatial_analysis=analysis,

            explanation=explanation
        )


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