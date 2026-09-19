# 🛰️ SATQuery — Model 2 API

### Bitemporal Satellite Image Change Detection

> **Model 2** is the change-detection engine of SATQuery.  
> It compares satellite images from two different points in time and identifies **where the scene has changed**, followed by spatial analysis and an explainable output.

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Model-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/BIT-Bitemporal%20Transformer-6C63FF?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LEVIR--CD-Baseline-8A2BE2?style=flat-square" />
  <img src="https://img.shields.io/badge/Change%20Detection-Binary-FF6B6B?style=flat-square" />
  <img src="https://img.shields.io/badge/Inference-256%C3%97256%20Patches-20B2AA?style=flat-square" />
</p>

---

## ✨ Overview

SATQuery Model 2 uses the **Bitemporal Image Transformer (BIT)** to perform pixel-level change detection between two temporally separated satellite images.

Instead of simply subtracting two images, BIT learns visual representations from both observations and models their relationships to identify meaningful changes.

### In simple terms:

```text
       BEFORE IMAGE                 AFTER IMAGE
            │                            │
            ▼                            ▼
     ┌──────────────┐            ┌──────────────┐
     │   Feature    │            │   Feature    │
     │   Encoder    │            │   Encoder    │
     └──────┬───────┘            └──────┬───────┘
            │                            │
            ▼                            ▼
        Feature                     Feature
         Tokens                       Tokens
            │                            │
            └────────────┬───────────────┘
                         ▼
                ┌─────────────────┐
                │   Transformer   │
                │    Encoder      │
                └────────┬────────┘
                         ▼
                   ┌───────────┐
                   │  Decoder  │
                   └─────┬─────┘
                         ▼
              Pixel-level Prediction
                         │
                         ▼
                  Binary Change Mask
