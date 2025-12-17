"""
Wrapper script to run weighted ensemble training
Run from project root: python run_weighted_ensemble.py
"""
import sys
from pathlib import Path

# Add credit_risk_fyp to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'credit_risk_fyp'))

# Import and run
from src.train_weighted_ensemble import main

if __name__ == '__main__':
    main()
