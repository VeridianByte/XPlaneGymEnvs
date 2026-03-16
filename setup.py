from setuptools import setup, find_packages
import os

# Read README as long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="XPlaneGym",
    version="0.1.0",
    description="X-Plane reinforcement learning environment compatible with OpenAI Gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Picaun",
    url="https://github.com/Picaun/XPlaneGym",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
        "stable_baselines3>=2.6.0",
        "tensorboard>=2.19.0"
    ],
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: Robot Framework :: Library",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Development Status :: 4 - Beta",
    ],
    keywords="reinforcement-learning, gymnasium, x-plane, flight-simulator, machine-learning, ai",
    python_requires=">=3.8",
    project_urls={
        "Bug Reports": "https://github.com/Picaun/XPlaneGym/issues",
        "Source": "https://github.com/Picaun/XPlaneGym"
    },
    include_package_data=True,
    license="MIT",
) 
