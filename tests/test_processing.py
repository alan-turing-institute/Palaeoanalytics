"""
Test the functions in read_and_process.py
"""
import pytest
import os
import yaml
from pylithics.src.read_and_process import read_image, detect_lithic, find_lithic_contours, process_image

import matplotlib.pyplot as plt


def test_read_image():

    filename = os.path.join('tests','test_images','RDK2_17_Dc_Pc_Lc.png')
    image_array, _ = read_image(filename)

    filename_tif = os.path.join('tests','test_images','2005_Erps-Kwerps-Villershof.tif')
    image_array_tif, _ = read_image(filename_tif)

    assert image_array.shape==(1595, 1465, 4)
    assert image_array_tif.shape==(445, 1548, 3)


def test_detect_lithic():

    filename = os.path.join('tests', 'test_images', 'RDK2_17_Dc_Pc_Lc.png')

    image_array, dpi = read_image(filename)

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)

    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

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

    assert binary_edge_sobel.shape==(1595,1465,4)

def test_find_lithic_contours():

    filename = os.path.join('tests', 'test_images', '8_test.png')

    image_array, dpi = read_image(filename)

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)

    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()

    image_processed = process_image(image_array, config_file)

    config_file['conversion'] = 0.1 #hardcoded for now
    binary_edge_sobel, _ = detect_lithic(image_processed, config_file)

    contours = find_lithic_contours(binary_edge_sobel, config_file)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)

    for contour, hierarchy, index in contours[['contour', 'hierarchy', 'index']].itertuples(index=False):
        try:
            if hierarchy[-1] == -1:
                linewidth = 3
                linestyle = 'solid'
                text = "Lithic"
            else:
                linewidth = 2
                linestyle = 'dashed'
                text = "Scar"

            ax.plot(contour[:, 0], contour[:, 1], linewidth=linewidth, linestyle=linestyle, label=text)
        except:
            continue

    fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')

    plt.figtext(0.02, 0.5, str(len(contours)) + ' contours')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join('tests', 'contour_detection_lithic.png'))
    plt.close(fig)

    print('Numer of contours:', contours[['contour']].shape[0])

    assert contours[['contour']].shape[0]!= 0


