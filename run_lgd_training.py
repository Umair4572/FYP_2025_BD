"""
Wrapper script to train LGD models
Run from project root: python run_lgd_training.py
"""
import sys
from pathlib import Path

# Add credit_risk_fyp to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'credit_risk_fyp'))

# Import and run
from src.train_lgd_model import main

if __name__ == '__main__':
    main()
