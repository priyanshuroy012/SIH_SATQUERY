from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

from app.services.satellite_service import (
    SatelliteService
)

from app.services.colab_bit_service import (
    ColabBITService
)

from app.services.change_analysis import (
    ChangeAnalyzer
)

from app.services.explanation_service import (
    ExplanationService
)


# =========================================================
# Project paths
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# =========================================================
# Application startup / shutdown
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting SATQuery Model 2 API")
    print("=" * 60)

    # -----------------------------------------------------
    # Reset application state
    # -----------------------------------------------------

    app.state.startup_error = None

    app.state.satellite_service = None
    app.state.colab_bit_service = None
    app.state.change_analyzer = None
    app.state.explanation_service = None

    try:

        # =================================================
        # 1. Initialize Satellite Service
        # =================================================

        print(
            "Initializing SatelliteService..."
        )

        satellite_service = (
            SatelliteService()
        )

        app.state.satellite_service = (
            satellite_service
        )

        print(
            "SatelliteService initialized."
        )


        # =================================================
        # 2. Initialize Colab BIT Service
        # =================================================

        print(
            "Initializing ColabBITService..."
        )

        colab_bit_service = (
            ColabBITService()
        )

        app.state.colab_bit_service = (
            colab_bit_service
        )

        print(
            "ColabBITService initialized."
        )


        # =================================================
        # 3. Check Colab BIT API
        # =================================================

        print(
            "Checking Colab BIT API..."
        )

        colab_health = (
            colab_bit_service.health_check()
        )

        print(
            f"Colab BIT API response:\n"
            f"{colab_health}"
        )

        print(
            "Colab BIT API is reachable."
        )


        # =================================================
        # 4. Initialize Change Analyzer
        # =================================================

        print(
            "Initializing ChangeAnalyzer..."
        )

        change_analyzer = (
            ChangeAnalyzer()
        )

        app.state.change_analyzer = (
            change_analyzer
        )

        print(
            "ChangeAnalyzer initialized."
        )


        # =================================================
        # 5. Initialize Explanation Service
        # =================================================

        print(
            "Initializing ExplanationService..."
        )

        explanation_service = (
            ExplanationService()
        )

        app.state.explanation_service = (
            explanation_service
        )

        print(
            "ExplanationService initialized."
        )


        # =================================================
        # 6. Startup complete
        # =================================================

        print("=" * 60)

        print(
            "SATQuery Model 2 initialized successfully."
        )

        print(
            "BIT inference is running on Google Colab."
        )

        print("=" * 60)


    except Exception as e:

        print("=" * 60)

        print(
            "ERROR: SATQuery Model 2 failed to initialize."
        )

        print("=" * 60)

        print(
            f"{type(e).__name__}: {e}"
        )

        app.state.startup_error = (
            f"{type(e).__name__}: {e}"
        )


    yield


    # =====================================================
    # Shutdown
    # =====================================================

    print("=" * 60)

    print(
        "Shutting down SATQuery Model 2 API..."
    )

    print("=" * 60)


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(

    title="SATQuery Model 2 API",

    description=(
        "Bitemporal satellite image change detection "
        "using the BIT model running on Google Colab."
    ),

    version="1.0.0",

    lifespan=lifespan
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# =========================================================
# API routes
# =========================================================

app.include_router(

    router,

    prefix="/api"
)


# =========================================================
# Root endpoint
# =========================================================

@app.get("/")
def root():

    return {

        "service": "SATQuery Model 2",

        "status": "running",

        "inference": "Google Colab",

        "message": (
            "Satellite change detection API is running."
        )
    }


# =========================================================
# Health endpoint
# =========================================================

@app.get("/health")
def health_check():

    satellite_service = getattr(
        app.state,
        "satellite_service",
        None
    )

    colab_bit_service = getattr(
        app.state,
        "colab_bit_service",
        None
    )

    change_analyzer = getattr(
        app.state,
        "change_analyzer",
        None
    )

    explanation_service = getattr(
        app.state,
        "explanation_service",
        None
    )

    startup_error = getattr(
        app.state,
        "startup_error",
        None
    )


    # -----------------------------------------------------
    # Determine service readiness
    # -----------------------------------------------------

    all_services_ready = all([

        satellite_service is not None,

        colab_bit_service is not None,

        change_analyzer is not None,

        explanation_service is not None
    ])


    if (
        all_services_ready
        and startup_error is None
    ):

        status = "healthy"

    else:

        status = "unhealthy"


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