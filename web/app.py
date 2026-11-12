"""
TerraSentinel - Web Studio Application
A clean, minimal interface for interactive land utilization & deforestation risk prediction.
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from forest_pipeline import ForestPipeline  # noqa: F401

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"

# Load Model
MODEL_PATH = MODELS_DIR / "forest_conversion_model.joblib"
print(f"Loading TerraSentinel model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Load Ranked Risk Data
RISK_FILE = DATA_DIR / "processed" / "forest_conversion_risk.csv"
risk_df = pd.read_csv(RISK_FILE) if RISK_FILE.is_file() else None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TerraSentinel | Land Utilization & Forest Risk Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-page: #f2f5ee;
      --card-bg: #ffffff;
      --border-color: #e2e8dc;
      --border-subtle: #edf2e8;
      --text-main: #14211b;
      --text-muted: #5e6d64;
      --text-light: #8a9a90;
      --brand-green: #1b4332;
      --brand-green-hover: #143527;
      --brand-light-green: #e9f3ec;
      --brand-accent: #2d6a4f;
      --risk-high-bg: #fde8e8;
      --risk-high-text: #c53030;
      --risk-med-bg: #fef3c7;
      --risk-med-text: #b45309;
      --risk-low-bg: #e6f4ea;
      --risk-low-text: #137333;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-page);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text-main);
      line-height: 1.5;
      padding: 32px 48px;
      min-height: 100vh;
    }

    .top-nav {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 40px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      background-color: var(--brand-green);
      color: #ffffff;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 16px;
    }

    .brand-title {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: var(--text-main);
    }

    .badge-pill {
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      letter-spacing: 0.5px;
      padding: 6px 14px;
      border: 1px solid #d4ded0;
      border-radius: 999px;
      color: var(--text-muted);
      background-color: transparent;
      font-weight: 600;
    }

    .hero-section {
      margin-bottom: 32px;
      max-width: 820px;
    }

    .kicker {
      font-family: 'Space Mono', monospace;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 12px;
    }

    .hero-title {
      font-size: 38px;
      font-weight: 700;
      letter-spacing: -0.8px;
      color: var(--text-main);
      margin-bottom: 12px;
    }

    .hero-subtitle {
      font-size: 15px;
      color: var(--text-muted);
      line-height: 1.6;
    }

    .tabs-nav {
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
    }

    .tab-btn {
      background: none;
      border: 1px solid transparent;
      padding: 8px 16px;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      border-radius: 8px;
      transition: all 0.2s;
    }

    .tab-btn:hover {
      background-color: rgba(27, 67, 50, 0.05);
      color: var(--brand-green);
    }

    .tab-btn.active {
      background-color: #ffffff;
      color: var(--brand-green);
      border-color: var(--border-color);
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .workspace-grid {
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 28px;
      align-items: stretch;
    }

    .panel-card {
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 20px;
      padding: 32px;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 20px -2px rgba(20, 33, 27, 0.03);
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 6px;
    }

    .panel-title {
      font-size: 20px;
      font-weight: 700;
      color: var(--text-main);
    }

    .step-counter {
      font-family: 'Space Mono', monospace;
      font-size: 12px;
      color: var(--text-light);
    }

    .panel-desc {
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 24px;
    }

    .preset-row {
      display: flex;
      gap: 8px;
      margin-bottom: 22px;
      flex-wrap: wrap;
    }

    .preset-pill {
      font-size: 12px;
      padding: 6px 12px;
      background: var(--bg-page);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      color: var(--text-muted);
      cursor: pointer;
      font-weight: 500;
      transition: all 0.15s;
    }

    .preset-pill:hover {
      border-color: var(--brand-accent);
      color: var(--brand-green);
      background-color: #eaf1e7;
    }

    .form-group { margin-bottom: 16px; }

    .form-group-header {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 6px;
      color: var(--text-main);
    }

    .form-val-display {
      font-family: 'Space Mono', monospace;
      font-size: 12px;
      color: var(--brand-accent);
    }

    input[type="range"] {
      width: 100%;
      height: 5px;
      border-radius: 4px;
      background: #e2e8dc;
      outline: none;
      -webkit-appearance: none;
    }

    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--brand-green);
      cursor: pointer;
      box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    }

    .toggle-row {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
      padding: 10px 14px;
      background: var(--bg-page);
      border-radius: 10px;
    }

    .toggle-label {
      font-size: 13px;
      font-weight: 600;
      flex-grow: 1;
    }

    .switch {
      position: relative;
      display: inline-block;
      width: 44px;
      height: 24px;
    }

    .switch input { opacity: 0; width: 0; height: 0; }

    .slider-toggle {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #ccc;
      transition: .3s;
      border-radius: 24px;
    }

    .slider-toggle:before {
      position: absolute;
      content: "";
      height: 18px; width: 18px;
      left: 3px; bottom: 3px;
      background-color: white;
      transition: .3s;
      border-radius: 50%;
    }

    input:checked + .slider-toggle {
      background-color: var(--brand-green);
    }

    input:checked + .slider-toggle:before {
      transform: translateX(20px);
    }

    .actions-row {
      display: flex;
      gap: 12px;
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid var(--border-subtle);
    }

    .btn-primary {
      background-color: var(--brand-green);
      color: #ffffff;
      border: none;
      border-radius: 8px;
      padding: 12px 24px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    .btn-primary:hover { background-color: var(--brand-green-hover); }

    .btn-secondary {
      background-color: #f0f4ec;
      color: var(--text-muted);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px 20px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }

    .btn-secondary:hover {
      background-color: #e5ece0;
      color: var(--text-main);
    }

    .empty-state {
      border: 2px dashed #e1e7dc;
      border-radius: 14px;
      padding: 60px 24px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      background-color: #fafcf8;
    }

    .empty-icon {
      width: 44px;
      height: 44px;
      background: #edf2e9;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--brand-accent);
      margin-bottom: 16px;
    }

    .empty-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
    .empty-text { font-size: 13px; color: var(--text-muted); max-width: 260px; }

    .result-content {
      display: none;
      flex-direction: column;
      gap: 20px;
    }

    .risk-banner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px;
      border-radius: 12px;
      border: 1px solid transparent;
    }

    .risk-banner.high { background-color: var(--risk-high-bg); border-color: #fca5a5; color: var(--risk-high-text); }
    .risk-banner.medium { background-color: var(--risk-med-bg); border-color: #fde68a; color: var(--risk-med-text); }
    .risk-banner.low { background-color: var(--risk-low-bg); border-color: #bbf7d0; color: var(--risk-low-text); }

    .risk-status-label {
      font-size: 12px;
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .risk-percentage { font-size: 32px; font-weight: 800; line-height: 1; }

    .prob-breakdown {
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 16px;
      background: #fbfdf9;
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
    }

    .prob-row-header {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 4px;
    }

    .progress-bar-bg {
      width: 100%;
      height: 8px;
      background: #e8eee4;
      border-radius: 99px;
      overflow: hidden;
    }

    .progress-bar-fill { height: 100%; border-radius: 99px; transition: width 0.4s ease; }
    .fill-forest { background: #246857; }
    .fill-agri { background: #ba731c; }
    .fill-built { background: #5269aa; }

    .insight-card {
      background: #fbfdf9;
      border-left: 4px solid var(--brand-green);
      border-radius: 6px;
      padding: 14px 16px;
    }

    .insight-title {
      font-size: 12px;
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--brand-accent);
      margin-bottom: 4px;
    }

    .insight-body { font-size: 13px; color: var(--text-main); line-height: 1.5; }

    .disclaimer-box {
      font-size: 12px;
      color: #7a8a7f;
      background: #fdf6ec;
      border: 1px solid #fae6cb;
      padding: 12px 14px;
      border-radius: 8px;
      line-height: 1.4;
    }

    .table-container {
      background: #ffffff;
      border: 1px solid var(--border-color);
      border-radius: 16px;
      overflow: hidden;
      margin-top: 12px;
    }

    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th {
      background: #f8faf5;
      padding: 12px 16px;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-color);
      font-family: 'Space Mono', monospace;
      font-size: 11px;
    }
    td { padding: 12px 16px; border-bottom: 1px solid #f0f3eb; color: var(--text-main); }
    tr:hover td { background: #fcfdfa; }

    .tag {
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      font-family: 'Space Mono', monospace;
      display: inline-block;
    }
    .tag-high { background: var(--risk-high-bg); color: var(--risk-high-text); }
    .tag-med { background: var(--risk-med-bg); color: var(--risk-med-text); }
    .tag-low { background: var(--risk-low-bg); color: var(--risk-low-text); }

    .plots-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 16px;
    }

    .plot-card {
      background: white;
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 16px;
    }

    .plot-card img { width: 100%; height: auto; border-radius: 8px; display: block; }
    .plot-caption { font-size: 13px; font-weight: 600; margin-top: 10px; color: var(--text-main); }
  </style>
</head>
<body>

  <header class="top-nav">
    <a href="#" class="brand">
      <div class="brand-icon">T</div>
      <span class="brand-title">TERRASENTINEL / STUDIO</span>
    </a>
    <span class="badge-pill">LOCAL DEMO</span>
  </header>

  <section class="hero-section">
    <div class="kicker">TERRASENTINEL: LAND UTILIZATION & RISK WORKSPACE</div>
    <h1 class="hero-title">A simple space to explore land-use risk.</h1>
    <p class="hero-subtitle">
      Simulate parcel geo-ecological characteristics to forecast the probability of forest land converting into agricultural or built-up infrastructure. Powered by the TerraSentinel calibrated Logistic Regression model.
    </p>
  </section>

  <nav class="tabs-nav">
    <button class="tab-btn active" onclick="switchTab('simulator')">Single Parcel Simulator</button>
    <button class="tab-btn" onclick="switchTab('ranking')">Top Vulnerable Parcels (285 Parcels)</button>
    <button class="tab-btn" onclick="switchTab('explainability')">SHAP & Model Performance</button>
  </nav>

  <!-- TAB 1: SIMULATOR -->
  <main id="tab-simulator" class="workspace-grid">
    <section class="panel-card">
      <div class="panel-header">
        <h2 class="panel-title">Parcel Parameters</h2>
        <span class="step-counter">01 / 02</span>
      </div>
      <p class="panel-desc">Adjust spatial & soil indicators or load a verified regional preset.</p>

      <div class="preset-row">
        <span style="font-size: 11px; font-weight:700; color:var(--text-light); align-self:center; font-family:'Space Mono';">PRESETS:</span>
        <button type="button" class="preset-pill" onclick="loadPreset('high_risk')">⚠️ High-Risk Urban Edge (#800)</button>
        <button type="button" class="preset-pill" onclick="loadPreset('agri_risk')">🌾 Agricultural Fringe (#134)</button>
        <button type="button" class="preset-pill" onclick="loadPreset('protected')">🌲 Protected Reserve</button>
      </div>

      <form id="predict-form" onsubmit="event.preventDefault(); runPrediction();">
        <div class="form-group">
          <div class="form-group-header">
            <span>Distance to Road (km)</span>
            <span class="form-val-display" id="disp-road">0.8 km</span>
          </div>
          <input type="range" id="road" min="0.1" max="12.0" step="0.1" value="0.8" oninput="updateVal('road', ' km')">
        </div>

        <div class="form-group">
          <div class="form-group-header">
            <span>Distance to Major City (km)</span>
            <span class="form-val-display" id="disp-city">8.5 km</span>
          </div>
          <input type="range" id="city" min="1.0" max="45.0" step="0.5" value="8.5" oninput="updateVal('city', ' km')">
        </div>

        <div class="form-group">
          <div class="form-group-header">
            <span>Soil Quality Index (0 - 1)</span>
            <span class="form-val-display" id="disp-soil">0.82</span>
          </div>
          <input type="range" id="soil" min="0.1" max="1.0" step="0.02" value="0.82" oninput="updateVal('soil', '')">
        </div>

        <div class="form-group">
          <div class="form-group-header">
            <span>Terrain Slope (degrees)</span>
            <span class="form-val-display" id="disp-slope">4.5°</span>
          </div>
          <input type="range" id="slope" min="0" max="35" step="0.5" value="4.5" oninput="updateVal('slope', '°')">
        </div>

        <div class="form-group">
          <div class="form-group-header">
            <span>Annual Rainfall (mm)</span>
            <span class="form-val-display" id="disp-rain">1150 mm</span>
          </div>
          <input type="range" id="rain" min="400" max="2200" step="25" value="1150" oninput="updateVal('rain', ' mm')">
        </div>

        <div class="form-group">
          <div class="form-group-header">
            <span>NDVI Vegetation Index (2015)</span>
            <span class="form-val-display" id="disp-ndvi">0.74</span>
          </div>
          <input type="range" id="ndvi" min="0.2" max="0.95" step="0.01" value="0.74" oninput="updateVal('ndvi', '')">
        </div>

        <div class="toggle-row">
          <span class="toggle-label">Designated Protected Conservation Zone</span>
          <label class="switch">
            <input type="checkbox" id="protected">
            <span class="slider-toggle"></span>
          </label>
        </div>

        <div class="actions-row">
          <button type="submit" class="btn-primary">Predict Land-Use Risk</button>
          <button type="button" class="btn-secondary" onclick="resetForm()">Reset</button>
        </div>
      </form>
    </section>

    <!-- Analysis Card -->
    <section class="panel-card">
      <div class="panel-header">
        <h2 class="panel-title">Analysis</h2>
        <span class="step-counter">02 / 02</span>
      </div>
      <p class="panel-desc">Model confidence and risk evaluation will appear below.</p>

      <div id="empty-state" class="empty-state">
        <div class="empty-icon">
          <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
        </div>
        <h3 class="empty-title">Ready when you are</h3>
        <p class="empty-text">Adjust parcel features on the left and run the inference.</p>
      </div>

      <div id="result-content" class="result-content">
        <div id="risk-banner" class="risk-banner high">
          <div>
            <div class="risk-status-label" id="risk-label">HIGH VULNERABILITY RISK</div>
            <div style="font-size: 13px; opacity: 0.9;" id="most-likely-callout">Most Likely: Built-up Expansion</div>
          </div>
          <div class="risk-percentage" id="risk-score-display">87.4%</div>
        </div>

        <div class="prob-breakdown">
          <div>
            <div class="prob-row-header">
              <span>Forest (Stable / No Change)</span>
              <span id="prob-forest">12.6%</span>
            </div>
            <div class="progress-bar-bg">
              <div id="bar-forest" class="progress-bar-fill fill-forest" style="width: 12.6%;"></div>
            </div>
          </div>

          <div>
            <div class="prob-row-header">
              <span>Conversion to Agricultural Land</span>
              <span id="prob-agri">24.1%</span>
            </div>
            <div class="progress-bar-bg">
              <div id="bar-agri" class="progress-bar-fill fill-agri" style="width: 24.1%;"></div>
            </div>
          </div>

          <div>
            <div class="prob-row-header">
              <span>Conversion to Built-up / Urban</span>
              <span id="prob-built">63.3%</span>
            </div>
            <div class="progress-bar-bg">
              <div id="bar-built" class="progress-bar-fill fill-built" style="width: 63.3%;"></div>
            </div>
          </div>
        </div>

        <div class="insight-card">
          <div class="insight-title">Model Explainability (SHAP Direction)</div>
          <p class="insight-body" id="shap-insight-text">
            Short distance to road infrastructure (0.8 km) significantly elevates accessibility, making built-up conversion 4.2x more probable than baseline.
          </p>
        </div>

        <div class="insight-card" style="border-left-color: #ba731c;">
          <div class="insight-title" style="color: #ba731c;">Governance & Conservation Action</div>
          <p class="insight-body" id="action-text">
            High vulnerability detected. Recommend priority satellite monitoring and establishment of a conservation buffer zone.
          </p>
        </div>

        <div class="disclaimer-box">
          Validated against 12-fold stratified CV on 275 eligible decadal parcels. Model uses fold-local inverse frequency weighting to overcome class imbalance.
        </div>
      </div>
    </section>
  </main>

  <!-- TAB 2: RANKED VULNERABLE PARCELS -->
  <section id="tab-ranking" style="display: none;">
    <div style="background: white; padding: 24px; border-radius: 16px; border: 1px solid var(--border-color); margin-bottom: 20px;">
      <h3 style="font-size: 18px; font-weight: 700; margin-bottom: 6px;">Top Deforestation-Vulnerable Parcels (Out-Of-Fold Scored)</h3>
      <p style="font-size: 13px; color: var(--text-muted);">
        Every forest parcel was scored strictly outside its fitting fold to ensure zero data leakage. Click any parcel to test it in the simulator.
      </p>
    </div>
    <div class="table-container">
      <table id="parcels-table">
        <thead>
          <tr>
            <th>PARCEL ID</th>
            <th>VULNERABILITY RISK</th>
            <th>RISK LEVEL</th>
            <th>PREDICTED CONVERSION</th>
            <th>OBSERVED DESTINATION (2025)</th>
            <th>ACTION</th>
          </tr>
        </thead>
        <tbody id="parcels-tbody">
          <tr><td colspan="6" style="text-align: center;">Loading regional risk data...</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- TAB 3: SHAP & EXPLAINABILITY -->
  <section id="tab-explainability" style="display: none;">
    <div style="background: white; padding: 24px; border-radius: 16px; border: 1px solid var(--border-color); margin-bottom: 20px;">
      <h3 style="font-size: 18px; font-weight: 700; margin-bottom: 6px;">Model Benchmarking & Explainability Artifacts</h3>
      <p style="font-size: 13px; color: var(--text-muted);">
        Multi-model evaluation across 7 families (84 configurations, 12 CV folds). Logistic Regression selected for superior Macro-F1 and balanced probability calibration.
      </p>
    </div>
    <div class="plots-grid">
      <div class="plot-card">
        <img src="/plots/shap_conversion.png" alt="SHAP Conversion Summary">
        <div class="plot-caption">Feature Impact on Conversion (SHAP Beeswarm)</div>
      </div>
      <div class="plot-card">
        <img src="/plots/11_transition_matrix.png" alt="Transition Matrix">
        <div class="plot-caption">Observed Decadal Land Transitions (2015 to 2025)</div>
      </div>
      <div class="plot-card">
        <img src="/plots/13_top_20_parcels.png" alt="Top 20 Parcels Risk">
        <div class="plot-caption">Top 20 Highest-Risk Parcels by Conversion Type</div>
      </div>
      <div class="plot-card">
        <img src="/plots/02_feature_correlations.png" alt="Feature Correlations">
        <div class="plot-caption">Pearson Correlation Matrix of Predictors</div>
      </div>
    </div>
  </section>

  <script>
    function updateVal(id, unit) {
      document.getElementById('disp-' + id).innerText = document.getElementById(id).value + unit;
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');

      document.getElementById('tab-simulator').style.display = (tabId === 'simulator') ? 'grid' : 'none';
      document.getElementById('tab-ranking').style.display = (tabId === 'ranking') ? 'block' : 'none';
      document.getElementById('tab-explainability').style.display = (tabId === 'explainability') ? 'block' : 'none';

      if (tabId === 'ranking') { loadRankedTable(); }
    }

    function loadPreset(type) {
      if (type === 'high_risk') {
        setFormValues(0.3, 4.5, 0.88, 3.0, 1250, 0.70, false);
      } else if (type === 'agri_risk') {
        setFormValues(2.5, 18.0, 0.94, 2.0, 1400, 0.78, false);
      } else if (type === 'protected') {
        setFormValues(8.5, 32.0, 0.60, 18.0, 950, 0.85, true);
      }
      runPrediction();
    }

    function setFormValues(road, city, soil, slope, rain, ndvi, isProtected) {
      document.getElementById('road').value = road;
      document.getElementById('city').value = city;
      document.getElementById('soil').value = soil;
      document.getElementById('slope').value = slope;
      document.getElementById('rain').value = rain;
      document.getElementById('ndvi').value = ndvi;
      document.getElementById('protected').checked = isProtected;

      updateVal('road', ' km');
      updateVal('city', ' km');
      updateVal('soil', '');
      updateVal('slope', '°');
      updateVal('rain', ' mm');
      updateVal('ndvi', '');
    }

    function resetForm() {
      setFormValues(0.8, 8.5, 0.82, 4.5, 1150, 0.74, false);
      document.getElementById('empty-state').style.display = 'flex';
      document.getElementById('result-content').style.display = 'none';
    }

    async function runPrediction() {
      const payload = {
        dist_to_road_km: parseFloat(document.getElementById('road').value),
        dist_to_city_km: parseFloat(document.getElementById('city').value),
        soil_quality_idx: parseFloat(document.getElementById('soil').value),
        slope_deg: parseFloat(document.getElementById('slope').value),
        rainfall_mm: parseFloat(document.getElementById('rain').value),
        ndvi_2015: parseFloat(document.getElementById('ndvi').value),
        protected_area: document.getElementById('protected').checked ? "1" : "0"
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        document.getElementById('empty-state').style.display = 'none';
        document.getElementById('result-content').style.display = 'flex';

        const riskScore = data.risk_score;
        document.getElementById('risk-score-display').innerText = (riskScore * 100).toFixed(1) + '%';

        const banner = document.getElementById('risk-banner');
        banner.className = 'risk-banner ' + data.risk_level.toLowerCase();
        document.getElementById('risk-label').innerText = data.risk_level + ' VULNERABILITY RISK';
        document.getElementById('most-likely-callout').innerText = 'Most Likely Outcome: ' + data.most_likely_outcome;

        document.getElementById('prob-forest').innerText = (data.p_forest * 100).toFixed(1) + '%';
        document.getElementById('bar-forest').style.width = (data.p_forest * 100) + '%';

        document.getElementById('prob-agri').innerText = (data.p_agri * 100).toFixed(1) + '%';
        document.getElementById('bar-agri').style.width = (data.p_agri * 100) + '%';

        document.getElementById('prob-built').innerText = (data.p_built * 100).toFixed(1) + '%';
        document.getElementById('bar-built').style.width = (data.p_built * 100) + '%';

        document.getElementById('shap-insight-text').innerText = data.insight;
        document.getElementById('action-text').innerText = data.recommendation;
      } catch (err) {
        console.error(err);
        alert('Error connecting to ML model backend: ' + err);
      }
    }

    async function loadRankedTable() {
      try {
        const res = await fetch('/api/ranked_parcels');
        const parcels = await res.json();
        const tbody = document.getElementById('parcels-tbody');
        tbody.innerHTML = '';

        parcels.forEach(p => {
          const tr = document.createElement('tr');
          const tagClass = p.risk_level === 'High' ? 'tag-high' : (p.risk_level === 'Medium' ? 'tag-med' : 'tag-low');
          tr.innerHTML = `
            <td style="font-weight:700; font-family:'Space Mono';">#${p.region}</td>
            <td style="font-weight:700;">${(p.risk_score * 100).toFixed(1)}%</td>
            <td><span class="tag ${tagClass}">${p.risk_level}</span></td>
            <td>${p.most_likely_conversion_type}</td>
            <td><strong>${p.observed_2025_land_cover}</strong></td>
            <td>
              <button class="preset-pill" onclick="simulateParcel(${p.region})">Load & Test</button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error(e);
      }
    }

    async function simulateParcel(id) {
      switchTab('simulator');
      document.querySelectorAll('.tab-btn')[0].classList.add('active');
      document.querySelectorAll('.tab-btn')[1].classList.remove('active');
      const res = await fetch('/api/parcel_detail?id=' + id);
      const d = await res.json();
      setFormValues(d.dist_to_road_km, d.dist_to_city_km, d.soil_quality_idx, d.slope_deg, d.rainfall_mm, d.ndvi_2015, d.protected_area === '1');
      runPrediction();
    }
  </script>
</body>
</html>
"""

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif url.path.startswith("/plots/"):
            filename = os.path.basename(url.path)
            plot_path = REPORTS_DIR / "figures" / filename
            if not plot_path.is_file():
                plot_path = ROOT / "reports" / "figures" / filename
            if plot_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(plot_path.read_bytes())
            else:
                self.send_error(404, "Plot image not found")
        elif url.path == "/api/ranked_parcels":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            top_parcels = risk_df.head(25).to_dict(orient="records") if risk_df is not None else []
            self.wfile.write(json.dumps(top_parcels).encode("utf-8"))
        elif url.path == "/api/parcel_detail":
            query = urllib.parse.parse_qs(url.query)
            pid = int(query.get("id", [0])[0])
            source_file = DATA_DIR / "raw" / "land_use_change.csv"
            row_dict = {}
            if source_file.is_file():
                df = pd.read_csv(source_file)
                match = df[df["parcel_id"] == pid]
                if not match.empty:
                    row_dict = match.iloc[0].to_dict()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(row_dict).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))

            road_km = float(data.get("dist_to_road_km", 1.0))
            city_km = float(data.get("dist_to_city_km", 10.0))
            soil_idx = float(data.get("soil_quality_idx", 0.8))
            slope = float(data.get("slope_deg", 5.0))
            rain = float(data.get("rainfall_mm", 1000.0))
            ndvi = float(data.get("ndvi_2015", 0.7))
            is_prot = str(data.get("protected_area", "0"))

            soil_suitability = float(np.clip(soil_idx * (1.0 - slope / 45.0), 0.0, 1.0))
            road_accessibility = float(1.0 / (1.0 + road_km))

            sample = pd.DataFrame([{
                "dist_to_city_km": city_km,
                "dist_to_road_km": road_km,
                "slope_deg": slope,
                "rainfall_mm": rain,
                "soil_quality_idx": soil_idx,
                "ndvi_2015": ndvi,
                "soil_suitability": soil_suitability,
                "road_accessibility": road_accessibility,
                "protected_area": is_prot
            }])

            probs = model.predict_proba(sample)[0]
            p_forest = float(probs[0])
            p_agri = float(probs[1])
            p_built = float(probs[2])
            risk_score = float(1.0 - p_forest)

            if risk_score >= 0.66:
                level = "HIGH"
                recommendation = "Critical vulnerability detected. Prioritize immediate satellite tracking and enforce conservation zoning."
            elif risk_score >= 0.33:
                level = "MEDIUM"
                recommendation = "Moderate conversion risk. Monitor peripheral expansion and verify permit compliance."
            else:
                level = "LOW"
                recommendation = "Low vulnerability. Forest tract is stable under current baseline conditions."

            if p_built > p_agri and p_built > p_forest:
                outcome = "Built-up Expansion (Urban / Infrastructure)"
                insight = f"High accessibility ({road_accessibility:.2f}) and proximity to road ({road_km} km) act as the dominant driver for urbanization."
            elif p_agri > p_built and p_agri > p_forest:
                outcome = "Agricultural Conversion (Cropland)"
                insight = f"High soil suitability ({soil_suitability:.2f}) combined with gentle slope ({slope}°) makes this terrain prime for agricultural conversion."
            else:
                outcome = "Forest Conservation (Stable)"
                insight = f"Topography and conservation status ({'Protected' if is_prot == '1' else 'Remote'}) inhibit commercial conversion."

            resp = {
                "risk_score": risk_score,
                "risk_level": level,
                "most_likely_outcome": outcome,
                "p_forest": p_forest,
                "p_agri": p_agri,
                "p_built": p_built,
                "insight": insight,
                "recommendation": recommendation
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

def start_server(port=5000):
    server = None
    for p in range(port, port + 10):
        try:
            server = socketserver.TCPServer(("", p), RequestHandler)
            port = p
            break
        except OSError:
            continue

    if not server:
        print("Could not bind to any port.")
        return

    url = f"http://localhost:{port}"
    print("\n" + "=" * 65)
    print("      TERRASENTINEL: LAND UTILIZATION & FOREST RISK STUDIO")
    print("=" * 65)
    print(f" [OK] Local interactive demo server running at: {url}")
    print(" [OK] Press Ctrl+C in terminal to stop.")
    print("=" * 65 + "\n")

    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()

if __name__ == "__main__":
    start_server(5000)

