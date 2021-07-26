"""
Test full pipeline
"""
from pylithics.src.read_and_process import read_image, detect_lithic
from pylithics.src.utils import mask_image
import os
import cv2
import numpy as np
import yaml

def test_mask_image():

    image_array = read_image(os.path.join('tests', 'test_images'), 'RDK2_17_Dc_Pc_Lc')


    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    _, contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[1]])

    masked_image = mask_image(image_array, cont)

    assert masked_image.shape != (0,0)
    assert masked_image.all()!=image_array.all()