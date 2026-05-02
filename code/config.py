from pathlib import Path

IMG_SIZE = (224, 224)
CANCER_TYPES = ["Brain", "Breast", "Skin", "Lung"]
MODELS_ROOT_DIR = Path("../models")

INVERT_OUTPUT_SCORE = False
THRESHOLD = 0.5