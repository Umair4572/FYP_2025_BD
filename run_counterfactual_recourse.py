"""
Wrapper script to run counterfactual recourse engine
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run counterfactual recourse
from credit_risk_fyp.src.counterfactual_recourse import main

if __name__ == '__main__':
    main()
