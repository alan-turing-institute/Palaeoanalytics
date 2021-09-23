from setuptools import setup

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name="pylithics",
    version="1.0",
    description="A lithics study using computer vision.",
    url="https://github.com/alan-turing-institute/Palaeoanalytics",
    author="Jason Gellis, Camila Rangel Smith",
    license="GNU GPLv3",
    include_package_data=True,
    packages=["pylithics",
              "pylithics.src",
               "pylithics.scripts"],
    install_requires=REQUIRED_PACKAGES,
    # we will need this later, i'll leave it commented as cont reminder.
    entry_points={"console_scripts": [
      "pylithics_run=pylithics.scripts.run:main",
      "pylithics_get_arrows=pylithics.scripts.run_arrows:main",

    ]},
)