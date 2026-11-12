"""
TerraSentinel - Model Prediction Engine
Provides single parcel and batch inference with automatic feature engineering.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd

# Ensure src is on path for ForestPipeline unpickling
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from forest_pipeline import ForestPipeline  # noqa: F401

DEFAULT_MODEL_PATH = ROOT / "models" / "forest_conversion_model.joblib"
DEFAULT_METADATA_PATH = ROOT / "models" / "forest_conversion_model_metadata.json"

class TerraSentinelPredictor:
    """Predictor for forest land-use conversion and vulnerability assessment."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
                 metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = joblib.load(self.model_path)
        
        if self.metadata_path.is_file():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {}

        self.class_labels = self.metadata.get("class_labels", ["no_change", "to_agricultural", "to_built_up"])

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive spatial accessibility and soil suitability heuristics."""
        df = df.copy()
        
        # Numeric conversions
        numeric_cols = ["dist_to_city_km", "dist_to_road_km", "slope_deg", "rainfall_mm", "soil_quality_idx", "ndvi_2015"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "dist_to_road_km" in df.columns and "road_accessibility" not in df.columns:
            df["road_accessibility"] = 1.0 / (1.0 + df["dist_to_road_km"])

        if "soil_quality_idx" in df.columns and "slope_deg" in df.columns and "soil_suitability" not in df.columns:
            df["soil_suitability"] = (df["soil_quality_idx"] * (1.0 - df["slope_deg"] / 45.0)).clip(0.0, 1.0)

        if "protected_area" in df.columns:
            df["protected_area"] = df["protected_area"].map(
                lambda v: str(int(v)) if pd.notna(v) and isinstance(v, (int, float)) and float(v).is_integer() else str(v)
            )

        feature_order = self.metadata.get("feature_order", [
            "dist_to_city_km", "dist_to_road_km", "slope_deg", "rainfall_mm",
            "soil_quality_idx", "ndvi_2015", "soil_suitability", "road_accessibility",
            "protected_area"
        ])
        return df[feature_order]

    def predict_parcel(self, parcel: Dict[str, Any]) -> Dict[str, Any]:
        """Predict conversion probabilities and vulnerability for a single parcel."""
        df = pd.DataFrame([parcel])
        X = self._engineer_features(df)
        probs = self.model.predict_proba(X)[0]

        p_forest = float(probs[0])
        p_agri = float(probs[1])
        p_built = float(probs[2])
        risk_score = float(1.0 - p_forest)

        if risk_score >= 0.66:
            risk_level = "HIGH"
            recommendation = "Critical vulnerability detected. Prioritize immediate satellite tracking and enforcement."
        elif risk_score >= 0.33:
            risk_level = "MEDIUM"
            recommendation = "Moderate conversion risk. Monitor peripheral expansion and verify zoning compliance."
        else:
            risk_level = "LOW"
            recommendation = "Low vulnerability. Forest tract is stable under baseline conditions."

        if p_built > p_agri and p_built > p_forest:
            outcome = "Built-up Expansion (Urban / Infrastructure)"
        elif p_agri > p_built and p_agri > p_forest:
            outcome = "Agricultural Conversion (Cropland)"
        else:
            outcome = "Forest Conservation (Stable)"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "most_likely_outcome": outcome,
            "probability_forest": p_forest,
            "probability_agricultural": p_agri,
            "probability_built_up": p_built,
            "recommendation": recommendation
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score an entire DataFrame of parcels."""
        X = self._engineer_features(df)
        probs = self.model.predict_proba(X)
        
        result = df.copy()
        result["risk_score"] = 1.0 - probs[:, 0]
        result["prob_forest_stable"] = probs[:, 0]
        result["prob_agricultural"] = probs[:, 1]
        result["prob_built_up"] = probs[:, 2]
        result["risk_level"] = np.where(result["risk_score"] >= 0.66, "High",
                               np.where(result["risk_score"] >= 0.33, "Medium", "Low"))
        return result

def main():
    print("=" * 60)
    print("       TERRASENTINEL: SINGLE PARCEL INFERENCE DEMO")
    print("=" * 60)
    
    predictor = TerraSentinelPredictor()

    sample = {
        "dist_to_city_km": 12.5,
        "dist_to_road_km": 0.3,
        "slope_deg": 4.2,
        "rainfall_mm": 1100.0,
        "soil_quality_idx": 0.85,
        "ndvi_2015": 0.72,
        "protected_area": "0"
    }

    print("\nInput Attributes:")
    for k, v in sample.items():
        print(f"  * {k:<20}: {v}")

    res = predictor.predict_parcel(sample)

    print("\n" + "-" * 60)
    print(f" Chance of Remaining Forest      : {res['probability_forest']:.1%}")
    print(f" Chance of Agricultural Expansion: {res['probability_agricultural']:.1%}")
    print(f" Chance of Built-up Expansion    : {res['probability_built_up']:.1%}")
    print("-" * 60)
    print(f" OVERALL VULNERABILITY RISK      : {res['risk_score']:.1%}")
    print(f" RISK LEVEL                      : [ {res['risk_level']} RISK ]")
    print(f" MOST LIKELY OUTCOME             : {res['most_likely_outcome']}")
    print(f" POLICY RECOMMENDATION           : {res['recommendation']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

