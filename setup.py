from setuptools import setup, find_packages

# Read the requirements.txt file for package dependencies
with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name="PyLithics",
    version="1.5",
    description="A Python package for stone tool analysis",
    download_url="https://github.com/alan-turing-institute/Palaeoanalytics/archive/refs/tags/v1.5.tar.gz",
    author="Jason Gellis, Camila Rangel Smith, Robert Foley",
    author_email='jg760@cam.ac.uk',
    license="GNU GPLv3",
    include_package_data=True,  # Ensures additional non-code files (e.g., data) are included
    packages=find_packages(),   # Automatically find packages in the current directory
    keywords=['lithics', 'human evolution', 'Archaeology', 'archeology',
              'lithic analysis', 'prehistoric technology', 'computer vision'],
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        "console_scripts": [
            "pylithics_run=pylithics.scripts.run:main",          # Command-line tool
            "pylithics_get_arrows=pylithics.scripts.run_arrows:main",  # Another command-line tool
        ],
    },
    long_description=open("README.md").read(),  # Optional, but recommended
    long_description_content_type="text/markdown",  # Optional, specify format if using README.md
)
