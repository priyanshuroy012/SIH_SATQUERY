import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.routes import router
from app.services.colab_bit_service import ColabBITService
from app.services.explanation_service import ExplanationService


# =========================================================
# Load environment variables
# =========================================================

load_dotenv()


# =========================================================
# Application startup / shutdown
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting SATQuery Model 2 API")
    print("=" * 60)

    # -----------------------------------------------------
    # Default state
    # -----------------------------------------------------

    app.state.startup_error = None

    app.state.colab_bit_service = None
    app.state.explanation_service = None

    try:

        # =================================================
        # 1. Check Colab BIT URL
        # =================================================

        colab_url = os.getenv("COLAB_BIT_URL")

        if not colab_url:

            raise RuntimeError(
                "COLAB_BIT_URL is not configured."
            )

        print(
            "Colab BIT API:"
        )

        print(
            colab_url
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
        # 3. Check Colab BIT health
        # =================================================

        print(
            "Checking Colab BIT API..."
        )

        health = (
            colab_bit_service.health_check()
        )

        print(
            f"Colab BIT health: {health}"
        )


        # =================================================
        # 4. Initialize Explanation Service
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
        # 5. Startup complete
        # =================================================

        print("=" * 60)

        print(
            "SATQuery Model 2 initialized successfully."
        )

        print(
            "BIT inference is running on Colab."
        )

        print("=" * 60)


    except Exception as e:

        # =================================================
        # Startup failure
        # =================================================

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


    # -----------------------------------------------------
    # Keep application running
    # -----------------------------------------------------

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
        "Satellite image change detection "
        "using the BIT model running on "
        "a remote Colab GPU."
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

        "bit_backend": "Google Colab",

        "message": (
            "Satellite change detection API is running."
        )
    }


# =========================================================
# Health endpoint
# =========================================================

@app.get("/health")
def health_check():

    colab_bit_service = getattr(
        app.state,
        "colab_bit_service",
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
    # Determine service state
    # -----------------------------------------------------

    colab_ready = (
        colab_bit_service is not None
    )

    explanation_ready = (
        explanation_service is not None
    )

    all_services_ready = (
        colab_ready
        and explanation_ready
    )


    if (
        all_services_ready
        and startup_error is None
    ):

        status = "healthy"

    else:

        status = "unhealthy"


    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return {

        "status": status,

        "service": "SATQuery Model 2",

        "bit_backend": "Google Colab",

        "services": {

            "colab_bit": (
                "connected"
                if colab_ready
                else "not_connected"
            ),

            "explanation": (
                "initialized"
                if explanation_ready
                else "not_initialized"
            )
        },

        "startup_error": startup_error
    }