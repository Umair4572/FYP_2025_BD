"""
Simple script to run the neural network training
Run this from the project root: python run_neural_network.py
"""
import sys
from pathlib import Path

# Add credit_risk_fyp to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'credit_risk_fyp'))

# Import and run the training script
from credit_risk_fyp.src.train_neural_network import *
