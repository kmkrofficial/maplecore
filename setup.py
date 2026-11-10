"""
MAPLE: Memory-Aware Predictive Loading Engine
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="maplecore",
    version="0.3.0",
    author="Keerthi Raajan K M",
    description="Experimental learned context selection for long-document LLM QA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kmkrofficial/maplecore",
    packages=find_packages(include=["maplecore", "maplecore.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "benchmarks": [
            "datasets>=2.19.0",
            "psutil>=5.9.0",
            "bm25s>=0.2.0",
            "rouge-score>=0.1.2",
            "nltk>=3.8",
            "einops>=0.7.0",
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0",
            "tqdm>=4.65.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "requests>=2.28.0",
            "rank_bm25>=0.2.2",
            "wandb>=0.15.0",
            "google-generativeai>=0.4.0",
            "python-dotenv>=1.0.0",
        ],
        "parsers": [
            "PyMuPDF>=1.23.0",
            "python-docx>=1.1.0",
            "beautifulsoup4>=4.12.0",
        ],
        "production": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "onnxscript>=0.1.0",
        ],
        "training": [
            "transformers>=4.45.0",
            "bitsandbytes>=0.45.0",
            "accelerate>=0.30.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "ruff>=0.5.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
)
