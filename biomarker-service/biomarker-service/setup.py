"""
Setup script for the RetinaXAI Biomarker Service.
"""

from setuptools import find_packages, setup

setup(
    name="retinaxai-biomarker-service",
    version="0.1.0",
    description="Isolated vascular biomarker extraction service for RetinaXAI",
    author="RetinaXAI Team",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.34.0",
        "httpx>=0.27.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "prometheus-client>=0.22.0",
        "loguru>=0.7.0",
        "numpy>=2.0.0",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.11.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "retinalysis-vascx>=0.5.0",
    ],
    entry_points={
        "console_scripts": [
            "biomarker-service=biomarker_service.main:main",
        ],
    },
    python_requires=">=3.10",
)
