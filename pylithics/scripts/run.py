#!/usr/bin/env python3
import argparse
import yaml
import os
import pandas as pd
from pylithics.src.read_and_process import read_image, find_lithic_contours, detect_lithic, process_image
from pylithics.src.plotting import plot_contours, plot_thresholding
from pylithics.src.utils import pixulator


def run_pipeline(id_list, metadata_df, input_dir, output_dir, config_file):
    """
    Script that runs the process of lithic characterisation on cont number of images.

    Parameters
    ----------
    id_list: list
        List of names identifying object present in cont directory
    metadata_df: dataframe

    input_dir: str
        path to input directory where images are found
    output_dir: str
        path to output directory to save outputs
    config_file: dict
        Dictionary with information of thresholding values

    """

    for id in id_list:

        config_file['id'] = id
        scale_id = metadata_df[metadata_df['PA_ID'] == id]['scale_ID'].values[0]

        if scale_id=="":
            continue

        scale_size = metadata_df[metadata_df['PA_ID'] ==id]['PA_scale'].values[0]

        config_file['scale_id'] = str(scale_id)
        config_file["scale_cm"] = scale_size

        run_characterisation(input_dir, output_dir, config_file)

    return 0


def run_characterisation(input_dir, output_dir, config_file):
    """
        Lithic characterisation of an image.

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

    name = os.path.join(input_dir,"images", id + '.png')
    name_scale = os.path.join(input_dir, "scales", config_file["scale_id"]+ '.png')

    image_array, image_pdi = read_image(name)
    image_scale_array, image_scale_dpi = read_image(name_scale)

    config_file['conversion_px']= pixulator(image_scale_array, config_file["scale_cm"])

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
    parser.add_argument('--metadata_filename', type=str, help='CSV file with information on the images and scale',
                        default="PA_DB.csv")

    args = parser.parse_args()
    filename_config = args.config

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.safe_load(config_file)

    images_input_dir = os.path.join(args.input_dir, "images")

    # path to the simulation files
    id_list = [i[:-4] for i in os.listdir(images_input_dir) if i.endswith('.png')]

    metadata_df = pd.read_csv(os.path.join(args.input_dir, args.metadata_filename), header = 0,
                              dtype = {'PA_ID': str,'scale_ID': str,'PA_scale': float}, engine = 'c')
    run_pipeline(id_list, metadata_df, args.input_dir, args.output_dir, config_file)


if __name__ == "__main__":
    main()
