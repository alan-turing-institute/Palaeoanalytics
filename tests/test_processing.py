"""
Test the functions in read_and_process.py
"""
import pytest
import os
import yaml
from pylithics.src.read_and_process import read_image, detect_lithic, find_lithic_contours

import matplotlib.pyplot as plt


def test_read_image():

    filename = os.path.join('tests','test_images','RDK2_17_Dc_Pc_Lc.png')
    image_array = read_image(filename)

    filename_tif = os.path.join('tests','test_images','2005_Erps-Kwerps-Villershof.tif')
    image_array_tif = read_image(filename_tif)

    assert image_array.shape==(1595,1465)
    assert image_array_tif.shape==(445,1548)


def test_detect_lithic():

    filename = os.path.join('tests', 'test_images', 'RDK2_17_Dc_Pc_Lc.png')

    image_array = read_image(filename)

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)

    binary_edge_sobel = detect_lithic(image_array, config_file['lithic'])

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                             figsize=(8, 8))
    axes = axes.ravel()

    axes[0].imshow(image_array, cmap=plt.cm.gray)
    axes[0].set_title('Original image')

    axes[1].imshow(binary_edge_sobel, cmap=plt.cm.gray)
    axes[1].set_title('Sobel Edge Detection')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('tests', 'edge_detection_lithic.png'))

    assert binary_edge_sobel.shape==(1595,1465)

def test_find_lithic_contours():

    filename = os.path.join('tests', 'test_images', 'RDK2_17_Dc_Pc_Lc_A.png')

    image_array = read_image(filename)

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)

    binary_edge_sobel = detect_lithic(image_array, config_file['lithic'])

    contours = find_lithic_contours(binary_edge_sobel, config_file['lithic'])

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(binary_edge_sobel, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join('tests', 'contour_detection_lithic.png'))

    print('Numer of contours:', len(contours))

    assert len(contours)== 0





