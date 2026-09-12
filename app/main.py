from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

from app.services.satellite_service import (
    SatelliteService
)

from app.services.bit_service import (
    BITService
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

# Project root:
#
# SIH/
# ├── BIT_CD/
# ├── app/
# └── venv/
#
BASE_DIR = Path(__file__).resolve().parent.parent


# BIT repository
BIT_REPO_PATH = (
    BASE_DIR / "BIT_CD"
)


# BIT pretrained checkpoint
BIT_CHECKPOINT_PATH = (
    BIT_REPO_PATH
    / "checkpoints"
    / "BIT_LEVIR"
    / "best_ckpt.pt"
)


# =========================================================
# Application startup / shutdown
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Starting SATQuery Model 2 API")
    print("=" * 60)


    try:

        # =================================================
        # 1. Check BIT repository
        # =================================================

        print(
            f"BIT repository:\n{BIT_REPO_PATH}"
        )

        if not BIT_REPO_PATH.exists():

            raise FileNotFoundError(
                "BIT_CD repository not found at:\n"
                f"{BIT_REPO_PATH}"
            )


        # =================================================
        # 2. Check BIT checkpoint
        # =================================================

        print(
            f"BIT checkpoint:\n"
            f"{BIT_CHECKPOINT_PATH}"
        )

        if not BIT_CHECKPOINT_PATH.exists():

            raise FileNotFoundError(
                "BIT checkpoint not found at:\n"
                f"{BIT_CHECKPOINT_PATH}"
            )


        # =================================================
        # 3. Initialize Satellite Service
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
        # 4. Initialize BIT Service
        # =================================================

        print(
            "Initializing BITService..."
        )

        bit_service = BITService(

            checkpoint_path=str(
                BIT_CHECKPOINT_PATH
            ),

            bit_repo_path=str(
                BIT_REPO_PATH
            ),

            patch_size=256,

            image_size=1024,

            device=None
        )

        app.state.bit_service = (
            bit_service
        )

        print(
            "BITService initialized."
        )


        # =================================================
        # 5. Initialize Change Analyzer
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
        # 6. Initialize Explanation Service
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
        # 7. Startup complete
        # =================================================

        print("=" * 60)
        print(
            "SATQuery Model 2 initialized successfully."
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

        # Keep the actual error available to the API
        app.state.startup_error = str(e)

        # Set services to None so /health can report
        # what failed.

        app.state.satellite_service = None
        app.state.bit_service = None
        app.state.change_analyzer = None
        app.state.explanation_service = None


    # Keep application running
    yield


    # =====================================================
    # Shutdown
    # =====================================================

    print("=" * 60)
    print("Shutting down SATQuery Model 2 API...")
    print("=" * 60)


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(

    title="SATQuery Model 2 API",

    description=(
        "Bitemporal satellite image change detection "
        "using the BIT model."
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

        "message": (
            "Satellite change detection API is running."
        )
    }


# =========================================================
# Health endpoint
# =========================================================

@app.get("/health")
def health_check():

    bit_service = getattr(

        app.state,

        "bit_service",

        None
    )

    satellite_service = getattr(

        app.state,

        "satellite_service",

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
        },

        "startup_error": getattr(
            app.state,
            "startup_error",
            None
        )
    }