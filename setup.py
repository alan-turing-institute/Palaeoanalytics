from setuptools import setup, find_packages

# Read the requirements.txt file for package dependencies
with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = [line.strip() for line in f.read().splitlines()
                        if line.strip() and not line.startswith('#')]

setup(
    name="PyLithics",
    version="2.0.0",  # Updated version
    description="A Python package for stone tool analysis with enhanced arrow detection and configuration management",
    download_url="https://github.com/alan-turing-institute/Palaeoanalytics/archive/refs/tags/v2.1.0.tar.gz",
    author="Jason Gellis, Camila Rangel Smith, Robert Foley",
    author_email='jg760@cam.ac.uk',
    license="GNU GPLv3",
    include_package_data=True,
    packages=find_packages(),
    package_data={
        'pylithics': ['config/*.yaml'],  # Include configuration files
    },
    keywords=['lithics', 'human evolution', 'Archaeology', 'archeology',
              'lithic analysis', 'prehistoric technology', 'computer vision',
              'arrow detection', 'DPI scaling'],  # Added new keywords
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        "console_scripts": [
            "pylithics_run=pylithics.app:main",  # Updated to point to enhanced app
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",  # Specify minimum Python version
)