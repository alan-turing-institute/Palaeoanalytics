"""Setup configuration for the PyLithics package."""
import os
from setuptools import find_namespace_packages, setup

# Get the current directory
here = os.path.abspath(os.path.dirname(__file__))

def read_requirements():
    """Return the runtime dependencies listed in requirements.txt."""
    with open(os.path.join(here, "requirements.txt"), "r", encoding="utf-8") as f:
        lines = (line.split('#', 1)[0].strip() for line in f)
        return [line for line in lines if line]


def read_long_description():
    """Return README.md as the long description."""
    with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
        return f.read()


# Record where this clone is, so the update check can read the branch
# and run the update in the right place. Ignored by git.
with open(os.path.join('pylithics', '_source_path.txt'), 'w') as _source:
    _source.write(os.path.dirname(os.path.abspath(__file__)))

setup(
    name="PyLithics",
    version="2.0.0",
    description="Quantitative analysis of stone tool drawings",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",

    # Author and contact information
    author="Jason Gellis, Camila Rangel Smith, Robert Foley",
    author_email="jg760@cam.ac.uk",

    # Project URLs
    url="https://github.com/alan-turing-institute/Palaeoanalytics",
    download_url=(
        "https://github.com/alan-turing-institute/Palaeoanalytics/archive/refs/tags/v2.0.0.tar.gz"
    ),
    project_urls={
        "Bug Reports": "https://github.com/alan-turing-institute/Palaeoanalytics/issues",
        "Source": "https://github.com/alan-turing-institute/Palaeoanalytics",
        "Documentation": "https://github.com/alan-turing-institute/Palaeoanalytics/wiki",
    },

    # License (SPDX) and classification
    license="GPL-3.0-or-later",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],

    # Package configuration. The data folders (config, models, data/pages)
    # have no __init__.py, so they are found as namespace packages and the
    # files to ship are listed in package_data.
    packages=find_namespace_packages(include=["pylithics", "pylithics.*"]),
    package_data={
        "pylithics": [
            "config/*.yaml",
            "config/*.yml",
            "data/pages/sample_plate.png",
            "models/*.pb",
            "models/*.json",
            "models/NOTICE",
            "_source_path.txt",
        ],
    },

    # Dependencies
    python_requires=">=3.10",
    install_requires=read_requirements(),

    # Optional dependencies for development
    extras_require={
        # Reading the identifiers printed beside each lithic on a plate
        # (pylithics-pages --read_labels). RapidOCR itself is installed
        # separately with --no-deps: it declares opencv-python by name,
        # which would reinstall the plain OpenCV build beside the contrib
        # build PyLithics needs. This extra carries its other dependencies.
        "ocr": ["onnxruntime>=1.14", "pyclipper>=1.3", "six>=1.16", "tqdm>=4.60"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "psutil>=5.9.0,<6.0.0",   # memory and file-handle checks in the tests
            "pyflakes>=3.0",
            "pydocstyle>=6.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "pymdown-extensions>=10.0",
        ],
    },

    # Console scripts
    entry_points={
        "console_scripts": [
            "pylithics=pylithics.app:main",
            "pylithics-pages=pylithics.page_segmentation.cli:main",
        ],
    },

    # Keywords for PyPI search
    keywords=[
        "lithics", "archaeology", "archeology", "human evolution",
        "lithic analysis", "prehistoric technology", "computer vision",
        "arrow detection", "DPI scaling", "image processing",
        "contour analysis", "geometric analysis", "stone tools",
        "page segmentation", "plate splitting", "figure extraction"
    ],

    # Zip safety
    zip_safe=False,
)