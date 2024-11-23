from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="retail-product-segmentation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A pipeline for retail product segmentation and semantic grouping using FastSAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/retail-product-segmentation",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.3",
        "matplotlib>=3.4.3",
        "fastsam>=0.1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "run_pipeline=src.pipeline:main",  # Assuming main pipeline logic is in src.pipeline.main
        ],
    },
)
