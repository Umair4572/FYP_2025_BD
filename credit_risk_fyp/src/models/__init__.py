"""Models package for Credit Risk Assessment FYP"""

from pathlib import Path

# Package version
__version__ = "1.0.0"

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
