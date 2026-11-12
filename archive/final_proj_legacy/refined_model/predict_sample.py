import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Ensure local modules (like ForestPipeline) are found
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    print("Loading trained model...")
    model_path = ROOT / "forest_conversion_model.joblib"
    model = joblib.load(model_path)

    # Example new parcel data
    sample = pd.DataFrame([{
        "dist_to_city_km": 12.5,
        "dist_to_road_km": 0.3,
        "slope_deg": 4.2,
        "rainfall_mm": 1100.0,
        "soil_quality_idx": 0.85,
        "ndvi_2015": 0.72,
        "soil_suitability": float(np.clip(0.85 * (1 - 4.2 / 45), 0, 1)),
        "road_accessibility": float(1 / (1 + 0.3)),
        "protected_area": "0"  # '0' for unprotected, '1' for protected
    }])

    print("\nInput Parcel Attributes:")
    for k, v in sample.iloc[0].items():
        print(f"  * {k}: {v}")

    # Predict probabilities
    probabilities = model.predict_proba(sample)[0]
    p_no_change = probabilities[0]
    p_agri = probabilities[1]
    p_built = probabilities[2]
    risk_score = 1.0 - p_no_change

    print("\n" + "=" * 45)
    print("      TERRASENTINEL: MODEL PREDICTION")
    print("=" * 45)
    print(f"Chance of Remaining Forest:        {p_no_change:.1%}")
    print(f"Chance of Agricultural Conversion: {p_agri:.1%}")
    print(f"Chance of Built-up Conversion:     {p_built:.1%}")
    print("-" * 45)
    print(f"OVERALL VULNERABILITY RISK:        {risk_score:.1%}")

    if risk_score >= 0.66:
        print("RISK LEVEL: [ HIGH RISK ] - Immediate intervention recommended.")
    elif risk_score >= 0.33:
        print("RISK LEVEL: [ MEDIUM RISK ] - Monitor land use.")
    else:
        print("RISK LEVEL: [ LOW RISK ] - Stable forest.")
    print("=" * 45)

if __name__ == "__main__":
    main()
