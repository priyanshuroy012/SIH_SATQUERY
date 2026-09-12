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

    bit_service = getattr(
        request.app.state,
        "bit_service",
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


    return {

        "status": "healthy",

        "service": "SATQuery Model 2",

        "services": {

            "satellite": (
                "initialized"
                if satellite_service is not None
                else "not_initialized"
            ),

            "bit": (
                "initialized"
                if bit_service is not None
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


    bit_service = getattr(

        request.app.state,

        "bit_service",

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


    if bit_service is None:

        raise HTTPException(

            status_code=500,

            detail=(
                "BIT service not initialized."
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
        # 2. Run BIT
        # =================================================

        print(
            "Running BIT change detection..."
        )


        prediction = (

            bit_service.detect(

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


        # =================================================
        # 3. Spatial analysis
        # =================================================

        print(
            "Running spatial analysis..."
        )


        analysis = (

            change_analyzer.analyze(

                prediction[
                    "change_mask"
                ]
            )
        )


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

                    data.before_date
                    .isoformat()
                ),

                after_date=(

                    data.after_date
                    .isoformat()
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

                    data.before_date
                    .isoformat()
                ),

                requested_after=(

                    data.after_date
                    .isoformat()
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

                    prediction[
                        "change_detected"
                    ]
                ),

                "change_percentage": (

                    prediction[
                        "change_percentage"
                    ]
                ),

                "changed_pixels": (

                    prediction[
                        "changed_pixels"
                    ]
                ),

                "total_pixels": (

                    prediction[
                        "total_pixels"
                    ]
                ),

                "image_size": (

                    prediction[
                        "image_size"
                    ]
                )
            },


            spatial_analysis=analysis,


            explanation=explanation
        )


    except HTTPException:

        # Preserve intentionally raised HTTP errors
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