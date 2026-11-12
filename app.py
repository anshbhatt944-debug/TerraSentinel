"""
TerraSentinel - Root launcher for the TerraSentinel Web Studio.
Run with: python app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from web.app import start_server

if __name__ == "__main__":
    start_server(5000)
