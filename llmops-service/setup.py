import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"
REPO_NAME = "retinaxai"
AUTHOR_USER_NAME = "louayamor"
SRC_REPO = "retinaxai-llmops"
AUTHOR_EMAIL = "amor.louay20@gmail.com"

setuptools.setup(
    
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="LLMOps service for report generation - RetinaXAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=setuptools.find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-text-splitters>=0.3.0",
        "langchain-community>=0.3.0",
        "langchain-huggingface>=0.1.0",
        "langchain-chroma>=0.1.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.0",
        "dagshub>=0.3.25",
        "mlflow>=2.7",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.7.4",
        "pydantic-settings>=2.2.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "azure-ai-inference>=1.0.0b9",
        "httpx>=0.27.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)