#!/usr/bin/env python3
import argparse
import yaml
import json
import os
import pandas as pd
from pylithics.src.read_and_process import read_image, find_lithic_contours, detect_lithic, process_image, data_output, get_arrows, read_arrow_data
from pylithics.src.plotting import plot_contours, plot_thresholding, plot_arrow_contours
from pylithics.src.utils import pixulator, find_arrow_templates, get_angles


def run_arrow_pipeline(id_list, input_dir, output_dir, config_file):
    """
    Script that runs the process of arrow extraction on a number of images.

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

        config_file['id'] = id

        # In this arrow extraction process we do not need to get scale measurements
        config_file['scale_id'] = "no scale"
        config_file['conversion_px'] = 1

        run_arrow_characterisation(input_dir, output_dir, config_file)

    return 0


def run_arrow_characterisation(input_dir, output_dir, config_file, debug=True):
    """
        Arrow extraction for an image.

        Parameters
        ----------
        input_dir: str
            path to input directory where image is found
        output_dir: str
            path to output directory to save outputs
        config_file: dict
            dictionary with information of thresholding values
        """
    id = config_file["id"]
    print('=============================')
    print('Processing figure: ', id)
    arrows = True

    name = os.path.join(input_dir, "images", id + '.png')
    image_array, image_pdi = read_image(name)

    image_processed = process_image(image_array, config_file)

    binary_array, threshold_value = detect_lithic(image_processed, config_file)

    if debug == True:
        output_threshold = os.path.join(output_dir, id + "_lithic_threshold.png")
        plot_thresholding(image_processed, threshold_value, binary_array, output_threshold)

    contours = find_lithic_contours(binary_array, config_file, arrows)

    index_drop, templates = find_arrow_templates(image_processed, contours)
    contours = contours[~contours['index'].isin(index_drop)]

    arrow_data_df = get_angles(templates, id)

    data_arrows_file = os.path.join('pylithics', "arrow_template_data","arrows" + id + ".pkl")

    arrow_data_df.to_pickle(data_arrows_file)

    output_lithic = os.path.join(output_dir, id + "_lithic_arrow_contours.png")
    plot_arrow_contours(image_array, contours, output_lithic)

    print('Arrow extraction for lithic '+id+' is done.')

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

    images_input_dir = os.path.join(args.input_dir, "images")

    # path to the simulation files
    id_list = [i[:-4] for i in os.listdir(images_input_dir) if i.endswith('.png')]

    run_arrow_pipeline(id_list, args.input_dir, args.output_dir, config_file)


if __name__ == "__main__":
    main()
