"""
Run the FastAPI server
"""
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import run

if __name__ == "__main__":
    run()
