#!/usr/bin/env python3
import argparse
import yaml
import os
from pylithics.src.read_and_process import read_image, find_lithic_contours, detect_lithic, \
    detect_scale, find_scale_contours

def run_pipeline(id_list, input_dir, output_dir, config_file):
    """
    Script that runs the process of lithic characterisation on a number of images.

    Parameters
    ----------
    id_list: list
        List of names identifying object present in a directory
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

    lithic_name = os.path.join(input_dir, id+"_lithic.png")

    lithic_image_array = read_image(lithic_name)

    binary_lithic_array = detect_lithic(lithic_image_array, config_file['lithic'])
    lithic_contours = find_lithic_contours(binary_lithic_array, config_file['lithic'])

    scale_name = os.path.join(input_dir,id+"_scale.png")
    scale_image_array = read_image(scale_name)

    binary_scale = detect_scale(scale_image_array, config_file['scale'])
    scale_contour = find_scale_contours(binary_scale, config_file['scale'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run lithics characterisation pipeline")

    parser.add_argument("-c", "--config", required=True, type=str, metavar="config-file",
                        help="the model config file (YAML)")
    parser.add_argument('--input_dir', help='directory where the input images are', default=None)
    parser.add_argument('--output_dir', type=str, help='directory where the output data is saved', default=None)

    args = parser.parse_args()
    filename_config = args.config

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)

    #id_scale.png
    #id_lithic.png

    # path to the simulation files
    id_list = [i.split('_')[0] for i in os.listdir(args.input_dir) if i.endswith('_scale.tiff')]

    run_pipeline(id_list, args.input_dir, args.output_dir, config_file)