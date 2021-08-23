#!/usr/bin/env python3
# install libraries for pylithics pipeline.
import argparse
import yaml
import json
import os
import pandas as pd
from pylithics.src.read_and_process import read_image,\
    find_lithic_contours, detect_lithic, process_image, data_output, \
    get_scars_angles, find_arrows
from pylithics.src.plotting import plot_results, plot_thresholding
from pylithics.src.utils import pixulator, get_angles


def run_pipeline(id_list, metadata_df, input_dir, output_dir, config_file, get_arrows):
    """
    Script that sets up directories, data identification, and configurations for lithic characterisation,
    and runs the pipeline.

    Parameters
    ----------
    id_list: list
        list of uniques identifier codes for images
    metadata_df: dataframe
        User defined table describing lithic images in id_list. Must include image file ID, scale ID, and scale measurement.
    input_dir: str
        path to input directory where images are found
    output_dir: str
        path to output directory to save processed images
    config_file: dict
        dictionary with information of thresholding values
    get_arrows: boolean
        Default = False. If True, PyLithics will find arrows and add them to the data.

    Returns
    -------
    none
    """

    for id in id_list:  # ID of individual lithic images
        config_file['id'] = id
        try:
            scale_id = metadata_df[metadata_df['PA_ID'] == id]['scale_ID'].values[0]  # ID of associated scale data
            if pd.isna(scale_id):
                print(
                    "Scale for Object " + id + " not available. No measurements will be calculated for this image.\
                    Results will be returned in pixels")
                config_file['scale_id'] = "no scale"
            else:
                scale_size = metadata_df[metadata_df['PA_ID'] == id]['PA_scale'].values[0]
                config_file['scale_id'] = str(scale_id)
                config_file["scale_mm"] = scale_size
        except (TypeError, IndexError):
            print("Scale ID and scale measurement for image " + id + " not found in metadata")
            print("No measurements will be calculated for this image")
            config_file['scale_id'] = "no scale"

        run_characterisation(input_dir, output_dir, config_file, get_arrows)

    return 0


def run_characterisation(input_dir, output_dir, config_file, arrows, debug=False):
    """
    Characterisation of image file.

    Parameters
    ----------
    input_dir: str
        path to input directory where images are found
    output_dir: str
        path to output directory to save processed images
    config_file: dict
        dictionary with information of thresholding values
    arrows: boolean
        If True, pylithics will collect templates for arrows.
    debug: flag to plot the outputs.

    Returns
    -------
    none
    """

    id = config_file["id"]
    print('=============================')
    print('Processing figure: ', id)

    # read image
    image_array = read_image(os.path.join(input_dir, 'images'), id)

    # get name of scale and if found read it
    try:
        image_scale_array = read_image(os.path.join(input_dir, "scales"), config_file["scale_id"])
        config_file['conversion_px'] = pixulator(image_scale_array, config_file["scale_mm"])
    except (FileNotFoundError):
        config_file['conversion_px'] = 1

    # initial processing of the image
    image_processed = process_image(image_array, config_file)

    # processing to detect lithic and scars
    binary_array, threshold_value = detect_lithic(image_processed, config_file)

    # show output of lithic detection for debugging
    if debug:
        output_threshold = os.path.join(output_dir, id + "_lithic_threshold.png")
        plot_thresholding(image_processed, threshold_value, binary_array, output_threshold)

    # find contours
    contours = find_lithic_contours(binary_array, config_file)

    # if this lithic has arrows do processing to detect and measure arrow angle
    if arrows:

        # get the templates for the arrows
        templates = find_arrows(image_array, image_processed, debug)

        # measure angles for existing arrows
        arrow_df = get_angles(templates)

        # associate arrows to scars, add that info into the contour
        contours = get_scars_angles(image_processed, contours, arrow_df)

    else:

        # if there is no arrows in the figures we can measure the angles differently
        contours = get_scars_angles(image_processed, contours)

    plot_results(id, image_array, contours, output_dir)

    # record and save data into a .json file
    json_output = data_output(contours, config_file)

    data_output_file = os.path.join(output_dir, id + ".json")
    print('Saving data to file: ', data_output_file)

    with open(data_output_file, 'w') as f:
        json.dump(json_output, f)

        print('Done.')


def main():
    parser = argparse.ArgumentParser(description="Run lithic characterisation pipeline")

    parser.add_argument("-c", "--config", required=True, type=str, metavar="config-file",
                        help="the model config file (YAML)")
    parser.add_argument('--input_dir', help='path to input directory where images are found', default=None)
    parser.add_argument('--output_dir', type=str, help='path to output directory to save processed image outputs',
                        default=None)
    parser.add_argument('--metadata_filename', type=str, help='CSV file with metadata on images and scales',
                        default=None)
    parser.add_argument('--get_arrows', action="store_true",
                        help='If a lithic contains arrows, find them and add them to the data',
                        default=False)

    args = parser.parse_args()
    filename_config = args.config

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.safe_load(config_file)

    images_input_dir = os.path.join(args.input_dir, "images")

    # path to the simulation files
    id_list = [i[:-4] for i in os.listdir(images_input_dir) if i.endswith('.png')]

    if args.metadata_filename == None:
        metadata_df = None
    else:
        metadata_df = pd.read_csv(os.path.join(args.input_dir, args.metadata_filename), header=0,
                                  dtype={'PA_ID': str, 'scale_ID': str, 'PA_scale': float}, engine='c')
    run_pipeline(id_list, metadata_df, args.input_dir, args.output_dir, config_file, args.get_arrows)


if __name__ == "__main__":
    main()
