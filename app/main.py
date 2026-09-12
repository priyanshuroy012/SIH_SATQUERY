from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

from app.services.colab_bit_service import ColabBITService
from app.services.change_analysis import ChangeAnalyzer
from app.services.explanation_service import ExplanationService


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting SATQuery Model 2...")

    app.state.colab_bit_service = None
    app.state.change_analyzer = None
    app.state.explanation_service = None
    app.state.startup_error = None

    try:

        # -----------------------------------------
        # Colab BIT
        # -----------------------------------------

        print("Initializing Colab BIT Service...")

        colab_bit_service = ColabBITService()

        print("Checking Colab BIT health...")

        health = colab_bit_service.health_check()

        print("Colab BIT health:", health)

        app.state.colab_bit_service = colab_bit_service

        # -----------------------------------------
        # Change Analyzer
        # -----------------------------------------

        print("Initializing ChangeAnalyzer...")

        app.state.change_analyzer = ChangeAnalyzer()

        # -----------------------------------------
        # Explanation Service
        # -----------------------------------------

        print("Initializing ExplanationService...")

        app.state.explanation_service = ExplanationService()

        print("SATQuery Model 2 initialized successfully.")

    except Exception as e:

        app.state.startup_error = f"{type(e).__name__}: {str(e)}"

        print("STARTUP ERROR:")
        print(app.state.startup_error)

    yield

    print("SATQuery Model 2 shutting down...")


app = FastAPI(
    title="SATQuery Model 2",
    description="Image-based bitemporal satellite change detection using BIT",
    version="2.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "service": "SATQuery Model 2",
        "status": "running",
        "input": "before_image + after_image"
    }


@app.get("/health")
def health():

    ready = all([
        app.state.colab_bit_service is not None,
        app.state.change_analyzer is not None,
        app.state.explanation_service is not None
    ])

    return {
        "status": "healthy" if ready else "unhealthy",
        "service": "SATQuery Model 2",
        "input_mode": "image_upload",
        "services": {
            "colab_bit": (
                "initialized"
                if app.state.colab_bit_service
                else "not_initialized"
            ),
            "change_analyzer": (
                "initialized"
                if app.state.change_analyzer
                else "not_initialized"
            ),
            "explanation": (
                "initialized"
                if app.state.explanation_service
                else "not_initialized"
            )
        },
        "startup_error": app.state.startup_error
    }