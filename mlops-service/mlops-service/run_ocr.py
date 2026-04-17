import os
from pathlib import Path

from app.domains.ocr.pipeline.ocr_pipeline import OCRPipeline
import sys


os.chdir(Path(__file__).parent)

if __name__ == "__main__":
    pipeline = OCRPipeline()
    reports = pipeline.run()
    sys.exit(0 if reports else 1)
