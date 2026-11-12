# TerraSentinel: AI-Powered Land Utilization & Deforestation Risk Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-EB5424)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blue)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()

> **TerraSentinel** is an end-to-end Machine Learning intelligence platform developed for **AI Sansad** to forecast changes in decadal land utilization, assess conversion likelihood of forest parcels into agricultural or urban infrastructure, and detect regions vulnerable to unsustainable land-use changes.

---

## Executive Overview

Predicting land-use conversion is critical for environmental governance, spatial planning, and proactive conservation enforcement. In real-world decadal satellite data, land transition is characterized by **extreme class imbalance**: the vast majority of forest land remains static, while high-consequence conversions into cropland or built-up infrastructure occur only in a small fraction of parcels.

**TerraSentinel** moves beyond naive vanity accuracy metrics by implementing a scientifically validated, leakage-free pipeline that balances rare-event detection with calibrated risk probabilities and explainability.

```
       [ Geo-Ecological Inputs ]
  (Road Distance, City Distance, Slope,
   Soil Quality, Rainfall, NDVI, Protection)
                   │
                   ▼
       [ Leakage-Free Preprocessing ]
  (Heuristic Soil Suitability, Road Accessibility,
         Inverse Frequency Power Weighting)
                   │
                   ▼
     [ TerraSentinel Calibrated Pipeline ]
         (Weighted Logistic Regression)
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[ Multi-Class Probabilities ]   [ Regional Vulnerability Screening ]
 • P(Forest Static)              • High Risk   (>= 66%) -> Urgent Intervention
 • P(To Agricultural)            • Medium Risk (33-66%) -> Boundary Monitoring
 • P(To Built-up)                • Low Risk    (< 33%)  -> Stable Conservation
```

---

## Key Achievements & Benchmark Results

We rigorously benchmarked **7 model families across 84 hyperparameter configurations and 12-fold stratified cross-validation** on 275 eligible decadal forest parcels.

### Held-Out Evaluation on 55 Completely Unseen Parcels

| Model Family | Cross-Val Macro F1 | Test Accuracy | Balanced Accuracy | Test Macro F1 | Macro OVR AUC | Rare Event Recall (Agri / Built-up) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TerraSentinel (Logistic Regression)** | **0.396** | **70.9% – 74.5%** | **43.5%** | **0.412** | **0.744** | **50.0% / 66.7%** |
| **Gradient Boosting** | 0.347 | 83.6% | 48.6% | 0.509 | 0.734 | 16.7% / 33.3% |
| **Extra Trees** | 0.328 | 67.3% | 56.5% | 0.470 | 0.783 | 66.7% / 33.3% |
| **Random Forest** | 0.326 | 81.8% | 43.0% | 0.436 | 0.708 | 0.0% / 33.3% |
| **XGBoost (Baseline)** | 0.357 | 81.8% | 43.0% | 0.410 | 0.734 | 0.0% / 33.3% |
| **Majority Rule (Always Forest)** | 0.304 | 83.6% | 33.3% | 0.304 | 0.500 | 0.0% / 0.0% |

### Why Weighted Logistic Regression Won over XGBoost:
1. **The Accuracy Trap Avoided**: An unweighted model achieves 83.6% accuracy by simply predicting "no change" every time—catching zero deforestation events. In fact, XGBoost suffered from 0.0% recall on agricultural conversion.
2. **Superior CV Macro-F1**: Across 12 folds, Logistic Regression achieved the highest CV Macro-F1 (**0.396** vs XGBoost's **0.357**).
3. **Smooth, Continuous Calibration**: For spatial risk ranking and vulnerability screening, monotonic log-odds probabilities provide far more reliable prioritization than the step-function probabilities produced by small-sample tree ensembles.

---

## Core Capabilities & Features

### 1. Interactive Studio Frontend (`app.py`)
A minimalist web studio matching modern design standards:
* **Real-Time Sliders**: Tweak road proximity, city distance, soil index, slope, rainfall, and NDVI with instant probability recalculation.
* **1-Click Policy Presets**: Load verified regional scenarios such as *Urban Sprawl Edge (#800)*, *Agricultural Fringe (#134)*, or *Protected Deep Forest*.
* **Interactive Parcel Explorer**: Search and load all 285 out-of-fold scored historical parcels.
* **Zero External Web Frameworks**: Runs natively via Python's standard library with zero overhead.

### 2. Production Drift & Data-Quality Monitor (`monitor.py`)
A command-line tool built for operational deployment:
* Computes **Histogram Total Variation Distance** on incoming parcel batches against decadal baseline distributions.
* Detects missing values and distribution shifts before scoring.
* Generates an executive breakdown of High, Medium, and Low risk parcels.

### 3. Model Explainability via SHAP & Permutation Analysis
* **Distance to Road & Accessibility**: Strongest overall driver of built-up urbanization.
* **Soil Suitability**: $( \text{soil\_quality} \times (1 - \text{slope} / 45) )$ is the primary catalyst for agricultural cropland conversion.
* **Protected Area Designation**: Unprotected status increases conversion likelihood by over **3.2x**.

---

## Project Structure

```
TerraSentinel/
├── app.py                                   # Root launcher for TerraSentinel Web Studio
├── land_use_change.csv                      # Full dataset (1,000 parcels, 12 attributes)
├── requirements.txt                         # Dependency specifications
├── README.md                                # Project documentation
├── .gitignore                               # Standard git ignore rules
│
└── final_proj/
    └── refined_model/                       # Production ML Model Package
        ├── app.py                           # Web Studio server & embedded frontend
        ├── forest_pipeline.py               # Custom ForestPipeline with sample weighting
        ├── monitor.py                       # Batch inference & data drift monitoring CLI
        ├── predict_sample.py                # Standalone single-parcel prediction script
        ├── monitoring_reference.json        # Baseline feature distributions & bin cuts
        ├── forest_conversion_model.joblib   # Trained & serialized production pipeline
        ├── forest_conversion_model_metadata.json # Complete metadata manifest
        ├── forest_conversion_risk.csv       # Out-of-fold risk scores for 285 parcels
        ├── test_held_out_55.csv             # 55 strictly isolated unseen test parcels
        ├── forest_conversion.ipynb          # End-to-end benchmark & analysis notebook
        ├── forest_conversion_final_report.md# Formal scientific evaluation report
        └── forest_conversion_results/
            ├── fitted_families/             # Saved joblib models for all 7 families
            ├── plots/                       # 14 publication-grade figures (SHAP, ROC, etc.)
            ├── held_out_model_comparison.csv# Benchmark metric matrix
            └── grid_search.csv              # Full 84-configuration hyperparameter search
```

---

## Quickstart Guide

### 1. Installation

Clone the repository and set up your Python environment (Python 3.10+ recommended):

```bash
git clone https://github.com/anshbhatt944-debug/TerraSentinel.git
cd TerraSentinel
pip install -r requirements.txt
```

### 2. Launch the Web Studio (Interactive UI)

Start the local interactive frontend:

```bash
python app.py
```
*Your browser will automatically open to `http://localhost:5000`.*

### 3. Run the Production Drift Monitor

Test the model against the 55 unseen, held-out test parcels:

```bash
cd final_proj/refined_model
python monitor.py test_held_out_55.csv
```

### 4. Run Single Parcel Prediction via CLI

```bash
cd final_proj/refined_model
python predict_sample.py
```

---

## Data Schema & Methodology

| Column Name | Type | Description / Handling |
| :--- | :---: | :--- |
| `parcel_id` | String | Unique spatial identifier (excluded from training). |
| `land_use_2015` | String | Baseline land cover in 2015 (used to filter 285 forest parcels). |
| `dist_to_road_km` | Float | Distance to transportation network in kilometers. |
| `dist_to_city_km` | Float | Distance to major urban center in kilometers. |
| `slope_deg` | Float | Terrain slope angle in degrees. |
| `rainfall_mm` | Float | Mean annual precipitation in millimeters. |
| `soil_quality_idx` | Float | Composite soil fertility index $[0, 1]$. |
| `ndvi_2015` | Float | Normalized Difference Vegetation Index in 2015. |
| `protected_area` | Binary | `1` if designated conservation zone, `0` if unprotected. |
| `population_growth_pct` | Float | **Excluded from training** due to undated timeframe to prevent look-ahead leakage. |
| `land_use_2025` | String | Target observation (Forest: 0, Cropland: 1, Built-up: 2). |

---

## Policy & Governance Impact (AI Sansad)

TerraSentinel transforms abstract satellite data into **actionable public policy**:
1. **Targeted Conservation Enforcement**: Out of 285 forest tracts, **39 High-Risk parcels** (e.g. Parcel `#910`, `#800`, `#898`) have been identified with over 85% conversion probability. Forest departments can proactively deploy drone patrols and boundary demarcations to these specific zones.
2. **Buffer Zone Planning**: By quantifying the exponential impact of road proximity ($1 / (1 + \text{dist\_to\_road})$), urban planners can mandate strict conservation easements along planned highway corridors.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
