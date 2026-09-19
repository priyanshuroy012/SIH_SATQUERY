SATQuery Model 2 API

Bitemporal Satellite Image Change Detection

Model 2 is the bitemporal change detection module of the SATQuery system. It compares two satellite images of the same geographical region captured at different points in time and identifies pixels/regions that have changed.

The API integrates a pretrained Bitemporal Image Transformer (BIT) with spatial analysis, visualization, and explanation services.

Model: BIT (Bitemporal Image Transformer)

Task: Binary change detection

Baseline dataset: LEVIR-CD

Inference: 256x256 patch-based processing

API: FastAPI

1. What the API Does

The API accepts:

Before image: earlier observation

After image: later observation

Processing flow:

Before Image ─┐
              ├──> BIT Change Detection
After Image ──┘          │
                         ↓
                  Change Prediction
                         │
                         ↓
                  Binary Change Mask
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
       Spatial Analysis        Visualization
              │                     │
              └──────────┬──────────┘
                         ↓
                    Explanation
                         ↓
                    API Response

The API reports:

Change detected or not

Change percentage

Changed pixels

Total pixels

Change severity

Number of connected change regions

Largest change region

Dominant spatial region

Binary change mask

Change overlay

Human-readable explanation

2. BIT Model

BIT = Bitemporal Image Transformer.

At a high level:

Before Image -> Feature Encoder -> Tokens ─┐
                                           ├-> Transformer -> Decoder
After Image  -> Feature Encoder -> Tokens ─┘
                                                        |
                                                        v
                                           Pixel-level change prediction

BIT learns relationships between two temporally separated images rather than relying only on direct pixel subtraction.

Why not simple image subtraction?

Pixel subtraction can be affected by:

Illumination differences

Shadows

Seasonal variation

Sensor noise

Image misregistration

Atmospheric conditions

A learned change-detection model can compare higher-level visual features.

3. Input Requirements

The API expects two corresponding images:

Parameter

Description

before_image

Earlier satellite image

after_image

Later satellite image

The images should represent approximately the same geographical area and preferably be spatially aligned.

The current pipeline processes images in 256x256 patches.

For a 1024x1024 image:

1024 / 256 = 4 patches per dimension

4 x 4 = 16 corresponding patch pairs

The patch predictions are reconstructed into the full-image change mask.

4. API Endpoints

Health Check

GET /health

Example:

{
  "status": "healthy",
  "service": "SATQuery Model 2",
  "services": {
    "colab_bit": "initialized",
    "change_analyzer": "initialized",
    "explanation": "initialized"
  }
}

Change Detection

POST /api/detect-change

Content type:

multipart/form-data

Parameters:

before_image: image file
after_image: image file

Example:

curl -X POST "http://localhost:8000/api/detect-change" ^
  -F "before_image=@before.png" ^
  -F "after_image=@after.png"

For Linux/macOS:

curl -X POST "http://localhost:8000/api/detect-change"   -F "before_image=@before.png"   -F "after_image=@after.png"

5. Response Structure

A successful response has the following general structure:

{
  "status": "success",
  "change_detection": {
    "change_detected": true,
    "change_percentage": 5.01,
    "changed_pixels": 52528,
    "total_pixels": 1048576,
    "image_size": [1024, 1024]
  },
  "spatial_analysis": {
    "change_percentage": 5.01,
    "severity": "moderate",
    "dominant_region": "central-right",
    "number_of_regions": 12,
    "largest_region_pixels": 18420
  },
  "visualizations": {
    "mask": "<base64-encoded PNG>",
    "overlay": "<base64-encoded PNG>"
  },
  "explanation": "Approximately 5.01% of the observed area shows detectable pixel-level change..."
}

Actual values depend on the submitted image pair.

6. Understanding the Outputs

Change Percentage

The API calculates:

changed pixels
---------------- x 100
total pixels

Change Mask

The model produces a prediction tensor. The API post-processes that prediction into a binary mask:

0 -> unchanged
1 -> changed

Overlay

The overlay is generated separately from the binary mask by highlighting detected changed pixels on the original/before image.

Therefore:

BIT prediction
      |
      v
Binary change mask
      |
      v
Overlay visualization

The overlay is a visualization of the prediction, not a direct BIT model output.

7. Spatial Analysis

The binary mask is further analyzed to provide interpretable spatial statistics.

Severity

The current application-level thresholds are:

Change percentage

Severity

0%

none

< 2%

low

< 10%

moderate

< 25%

high

>= 25%

very high

These are application rules, not universal remote-sensing standards.

Connected Regions

Connected-component analysis groups spatially connected changed pixels.

The API reports:

number_of_regions

largest_region_pixels

Dominant Region

The largest connected region is located using its centroid. The image is conceptually divided into a 3x3 spatial grid:

+-----------+-----------+-----------+
| upper     | upper     | upper     |
| left      | central   | right     |
+-----------+-----------+-----------+
| central   | central   | central   |
| left      | central   | right     |
+-----------+-----------+-----------+
| lower     | lower     | lower     |
| left      | central   | right     |
+-----------+-----------+-----------+

This is a spatial description, not semantic classification.

8. Explanation Layer

The explanation service converts computed statistics into a human-readable summary.

Example:

Approximately 5.01% of the observed area shows detectable
pixel-level change. The overall change severity is moderate.
The detected changes are primarily concentrated in the
central-right portion of the scene.

The current Model 2 does not independently determine the semantic cause of change.

For example, it should not automatically claim:

"New construction was detected."

or:

"Vegetation loss occurred."

without an additional semantic model.

A useful distinction is:

BIT answers:       WHERE did change occur?
Semantic model:    WHAT caused the change?

9. Project Structure

A typical Model 2 backend is organized approximately as:

app/
├── main.py
├── routes/
│   └── change_detection.py
├── services/
│   ├── colab_bit_service.py
│   ├── change_analyzer.py
│   └── explanation_service.py
└── ...

main.py

Initializes the FastAPI application and required services.

BIT service

Handles BIT inference and communication with the GPU inference environment.

Change analyzer

Performs:

Change percentage calculation

Severity classification

Connected-component analysis

Largest-region detection

Dominant-region calculation

Explanation service

Generates a human-readable explanation from the spatial analysis.

10. Running Locally

Create a virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Linux/macOS:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Start FastAPI:

uvicorn app.main:app --reload

The API will be available at:

http://localhost:8000

Interactive Swagger documentation:

http://localhost:8000/docs

OpenAPI specification:

http://localhost:8000/openapi.json

11. GPU Inference

The current prototype uses a remote GPU environment for BIT inference.

Architecture:

Frontend
   |
   v
FastAPI Backend
   |
   v
Colab BIT Service
   |
   v
GPU Inference
   |
   v
FastAPI Backend
   |
   v
Frontend

A Google Colab T4 GPU can be used during development.

The Colab service can be exposed through a temporary tunnel such as ngrok.

Important: Colab + ngrok is a prototype/demo deployment approach, not a production architecture. A production deployment should use a persistent GPU inference service.

12. Model Configuration

The implementation uses the BIT LEVIR-CD configuration:

base_transformer_pos_s4_dd8_dedim8

The corresponding pretrained checkpoint is:

best_ckpt.pt

The SATQuery integration uses the pretrained model for inference rather than claiming to have trained BIT from scratch.

13. Dataset

The baseline model uses LEVIR-CD, a remote-sensing change-detection dataset containing:

637 image pairs

1024x1024 image patches

Very-high-resolution imagery

Building-related changes

Multi-year temporal observations

Sentinel-2 Consideration

LEVIR-CD and Sentinel-2 are different imaging domains.

Important differences include:

Sensor characteristics

Spatial resolution

Spectral information

Image appearance

Geographic distribution

Therefore, a LEVIR-CD pretrained BIT model should be treated as a baseline when applied to Sentinel-2 imagery. Sentinel-2-specific fine-tuning or domain adaptation is required for stronger deployment claims.

14. Evaluation

Accuracy alone can be misleading in change detection because unchanged pixels may dominate the image.

Useful metrics include:

Precision

Recall

F1-score

IoU

Specificity

F1-score

              2 x Precision x Recall
F1 = -----------------------------------------
          Precision + Recall

IoU

                 Intersection
IoU = --------------------------------
       Union of prediction and truth

These metrics provide more useful information about the detected change regions than accuracy alone.

15. End-to-End Processing

1. User uploads Before image
              +
2. User uploads After image
              |
              v
3. FastAPI receives image pair
              |
              v
4. Images are validated
              |
              v
5. Images are divided into corresponding 256x256 patches
              |
              v
6. BIT processes each temporal patch pair
              |
              v
7. Patch predictions are reconstructed
              |
              v
8. Binary change mask is generated
              |
              v
9. Change percentage is calculated
              |
              v
10. Connected regions are identified
              |
              v
11. Severity and dominant region are calculated
              |
              v
12. Overlay is generated
              |
              v
13. Explanation is generated
              |
              v
14. JSON response is returned to frontend

16. Development Contribution

This repository represents the Model 2 API/integration layer of SATQuery.

The project does not claim to invent the BIT architecture. The SATQuery contribution is the surrounding system integration, including:

BIT inference integration

Patch-based inference

Full-image mask reconstruction

Spatial change quantification

Connected-region analysis

Change visualization

Human-readable explanation

FastAPI integration

GPU inference integration

17. Limitations

Current limitations include:

Domain gap between LEVIR-CD and Sentinel-2 imagery

Binary change detection rather than semantic change classification

Sensitivity to clouds, shadows, seasonal effects, and image misregistration

Colab/ngrok-based inference is not production-ready

Current dominant-region output is spatial, not semantic

18. Future Improvements

Potential improvements:

Sentinel-2-specific fine-tuning

Domain adaptation

Cloud and image-quality filtering

Improved image registration

Semantic change classification

Confidence estimation

Persistent GPU deployment

More robust temporal alignment

Evaluation across geographically diverse datasets

19. References

BIT

Remote Sensing Image Change Detection with Transformers

Official repository:

https://github.com/justchenhao/BIT_CD

Paper:

https://arxiv.org/abs/2103.00208

LEVIR-CD

Official repository:

https://github.com/justchenhao/LEVIR

20. Usage Notice

The BIT repository and pretrained model are subject to the licensing and usage conditions specified by the original authors.

Refer to the official BIT repository for the applicable research/non-commercial and commercial-use conditions.

SATQuery Model 2

Natural Language Query
        |
        v
     SATQuery
        |
   +----+----+----------------+
   |         |                |
   v         v                v
 VQA      Change          Temporal
          Detection       Intelligence
            Model 2
              |
              v
     Evidence + Spatial Analysis
              |
              v
          Final Answer
