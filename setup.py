from setuptools import setup

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name="PyLithics",
    version="1.0",
    description="A Python package for stone tool analysis",
    download_url="https://github.com/alan-turing-institute/Palaeoanalytics",
    author="Jason Gellis, Camila Rangel Smith, Robert Foley",
    author_email='jg760@cam.ac.uk',
    license="GNU GPLv3",
    include_package_data=True,
    packages=["pylithics",
              "pylithics.src",
              "pylithics.scripts"],
    keywords = ['lithics', 'human evolution', 'Archaeology', 'archeology',
                'lithic analysis', 'prehistoric technology', 'computer vision'],
    install_requires=REQUIRED_PACKAGES,
    # we will need this later, I'll leave it commented as cont reminder.
    entry_points={"console_scripts": [
      "pylithics_run=pylithics.scripts.run:main",
      "pylithics_get_arrows=pylithics.scripts.run_arrows:main",

    ]},
)
