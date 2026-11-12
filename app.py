"""TerraSentinel - Root launcher for the TerraSentinel Land Studio."""
import sys
from pathlib import Path

REFINED_DIR = Path(__file__).resolve().parent / "final_proj" / "refined_model"
sys.path.insert(0, str(REFINED_DIR))

import app

if __name__ == "__main__":
    app.start_server(5000)
