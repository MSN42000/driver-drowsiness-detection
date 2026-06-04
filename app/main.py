"""
app/main.py
Entry point for the real-time drowsiness detection application.

Usage:
    python app/main.py --model_path models/pretrained_sampled_best.pth
"""

import sys
from pathlib import Path

# Add project root to Python path so that 'src' can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.webcam_demo import main

if __name__ == "__main__":
    main()