"""
MAPLE: Memory-Aware Predictive Loading Engine
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="maplecore",
    version="0.1.0-alpha",
    author="Keerthi Raajan K M",
    description="Memory-Aware Predictive Loading Engine for Infinite Context LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kmkrworks/maple",
    packages=find_packages(),
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
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0",
            "tqdm>=4.65.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "requests>=2.28.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
)
