#!/usr/bin/env python3
import argparse
import yaml
import os
from pylithics.src.read_and_process import read_image, find_lithic_contours, detect_lithic, process_image

from pylithics.src.plotting import plot_contours, plot_thresholding


def run_pipeline(id_list, input_dir, output_dir, config_file):
    """
    Script that runs the process of lithic characterisation on cont number of images.

    Parameters
    ----------
    id_list: list
        List of names identifying object present in cont directory
    input_dir: str
        path to input directory where images are found
    output_dir: str
        path to output directory to save outputs
    config_file: dict
        Dictionary with information of thresholding values

    """

    for id in id_list:
        run_characterisation(id, input_dir, output_dir, config_file)

    return 0


def run_characterisation(id, input_dir, output_dir, config_file):
    """
        Lithic characterisation of an image.

        Parameters
        ----------
        id: string
            Names identifying object
        input_dir: str
            path to input directory where image is found
        output_dir: str
            path to output directory to save outputs
        config_file: dict
            dictionary with information of thresholding values
        """

    # start with lithic
    print('=============================')
    print('Processing figure: ', id)

    name = os.path.join(input_dir, id)

    image_array, image_pdi = read_image(name)

    image_processed = process_image(image_array, config_file)

    binary_array, threshold_value = detect_lithic(image_processed, config_file)

    output_threshold = os.path.join(output_dir, id + "_lithic_threshold.png")
    plot_thresholding(image_processed, threshold_value, binary_array, output_threshold)

    contours = find_lithic_contours(binary_array, config_file)

    output_lithic = os.path.join(output_dir, id + "_lithic_contours.png")
    plot_contours(image_array, contours, output_lithic)


    print('Done.')


def main():
    parser = argparse.ArgumentParser(description="Run lithics characterisation pipeline")

    parser.add_argument("-c", "--config", required=True, type=str, metavar="config-file",
                        help="the model config file (YAML)")
    parser.add_argument('--input_dir', help='directory where the input images are', default=None)
    parser.add_argument('--output_dir', type=str, help='directory where the output data is saved', default=None)

    args = parser.parse_args()
    filename_config = args.config

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.safe_load(config_file)

    # id_scale.png
    # id_lithic.png

    # path to the simulation files
    id_list = [i for i in os.listdir(args.input_dir) if i.endswith('.png')]

    run_pipeline(id_list, args.input_dir, args.output_dir, config_file)

if __name__ == "__main__":
    main()