"""
Setup configuration for PyLithics package.
"""
import os
from setuptools import setup, find_packages

# Get the current directory
here = os.path.abspath(os.path.dirname(__file__))

# Read the requirements.txt file for package dependencies
def read_requirements():
    requirements_path = os.path.join(here, "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and built-in modules
                if line and not line.startswith('#') and not line.startswith('# '):
                    requirements.append(line)
            return requirements
    else:
        # Fallback requirements if file doesn't exist
        return [
            "opencv-python-headless>=4.8.0,<5.0.0",
            "Pillow>=10.0.0,<11.0.0",
            "numpy>=1.24.0,<2.0.0",
            "PyYAML>=6.0,<7.0",
            "pandas>=1.5.0,<3.0.0",
            "scipy>=1.9.0,<2.0.0",
            "shapely>=2.0.0,<3.0.0",
            "matplotlib>=3.6.0,<4.0.0",
            "setuptools>=65.0.0",
            "wheel>=0.38.0",
        ]

# Read the long description from README
def read_long_description():
    readme_path = os.path.join(here, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        PyLithics: Professional Stone Tool Analysis Software
        ====================================================

        PyLithics is a comprehensive archaeological tool for analyzing lithic artifacts
        using computer vision and advanced image processing techniques.

        Features:
        - Automated contour detection and geometric analysis
        - Advanced arrow detection with DPI-aware scaling
        - Surface classification (Dorsal, Ventral, Platform, Lateral)
        - Symmetry analysis for dorsal surfaces
        - Voronoi diagram spatial analysis
        - Comprehensive metric calculation and CSV export
        """

setup(
    name="PyLithics",
    version="2.0.0",
    description="A Python package for stone tool analysis with enhanced arrow detection and configuration management",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",

    # Author and contact information
    author="Jason Gellis, Camila Rangel Smith, Robert Foley",
    author_email="jg760@cam.ac.uk",

    # Project URLs
    url="https://github.com/alan-turing-institute/Palaeoanalytics",
    download_url="https://github.com/alan-turing-institute/Palaeoanalytics/archive/refs/tags/v2.0.0.tar.gz",
    project_urls={
        "Bug Reports": "https://github.com/alan-turing-institute/Palaeoanalytics/issues",
        "Source": "https://github.com/alan-turing-institute/Palaeoanalytics",
        "Documentation": "https://github.com/alan-turing-institute/Palaeoanalytics/wiki",
    },

    # License and classification
    license="GNU GPLv3",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],

    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pylithics": [
            "config/*.yaml",
            "config/*.yml",
            "data/examples/*",
            "templates/*",
        ],
    },

    # Dependencies
    python_requires=">=3.8",
    install_requires=read_requirements(),

    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },

    # Console scripts
    entry_points={
        "console_scripts": [
            "pylithics=pylithics.app:main",
            "pylithics-run=pylithics.app:main",  # Alternative name
        ],
    },

    # Keywords for PyPI search
    keywords=[
        "lithics", "archaeology", "archeology", "human evolution",
        "lithic analysis", "prehistoric technology", "computer vision",
        "arrow detection", "DPI scaling", "image processing",
        "contour analysis", "geometric analysis", "stone tools"
    ],

    # Zip safety
    zip_safe=False,
)