"""
Wrapper script to run credit risk stress testing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run stress testing
from credit_risk_fyp.src.stress_testing import main

if __name__ == '__main__':
    main()
