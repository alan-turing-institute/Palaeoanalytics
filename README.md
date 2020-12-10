# Palaeoanalytics
Repository for the Paleoanalytics project.

# Installation

The `pylithics` package requires Python 3.6 or greater. To install, start by creating a fresh conda environment.
```
conda create -n paleo python=3.7
conda activate paleo
```

Get the source.
```
git clone https://github.com/alan-turing-institute/Palaeoanalytics.git
```

Enter the repository and check out a relevant branch if necessary (the develop branch contains the most up to date stable version of the code).
```
cd Palaeoanalytics
git checkout develop
```
Install the package using `pip`.
```
pip install .
```

# Running pylithics


*pylithics* can be run via command line. The following command displays all available options:

```bash
pylithics_run --help
```

Output:

```bash
usage: pylithics_run [-h] -c config-file [--input_dir INPUT_DIR]
                     [--output_dir OUTPUT_DIR]

Run lithics characterisation pipeline

optional arguments:
  -h, --help            show this help message and exit
  -c config-file, --config config-file
                        the model config file (YAML)
  --input_dir INPUT_DIR
                        directory where the input images are
  --output_dir OUTPUT_DIR
                        directory where the output data is saved

```

For example, given that you have a set of lithics images (and it respective scales), you can run the pylithics processing script with the
following:

```
pylithics_run -c configs/test_config.yml --input_dir <path_to_input_images_dir> --output_dir <path_to_output_directory>
```

This ```test_config.yml``` config file contains the following options:


```yaml
lithic:
  threshold: 0.95
  contour_parameter: 0.4
  contour_fully_connected: 'high'
  minimum_pixels_contour: 500
scale:
  threshold: 1.0
  contour_parameter: 0.50
  contour_fully_connected: 'high'
  minimum_pixels_contour: 10
```

You can modify or create your on config file and provide it to the CLI. 

The images found in ```<path_to_input_images_dir>``` must follow the this nomenclature:

```bash
input_directory
   ├── id1_lithics.png
   ├── id1_scale.png
   ├── id2_lithics.png
   ├── id2_scale.png
   ├── id3_lithics.png
   └── id3_scale.png
   ├── ...
   ├── idn_lithics.png
   └── idn_scale.png
```

where each pair of lithics-scale images has an unique identifier (id1, id2, ..., idn). For the moment we only accept *png* images. 

