import matplotlib.pyplot as plt
import numpy as np
import math
import pylithics.src.utils as utils


def plot_contours(image_array, contours, output_path):
    """
    Plot the results of the lithic object characterisation.

    Parameters
    ----------
    image_array: array
        array of an unprocessed image read by openCV (0, 255 pixels)
    contours: dataframe
        pixel coordinates for a contour of a single flake scar ############
        or outline of a lithic object detected by contour finding
    output_path:
        path to output directory to save processed images

    Returns
    -------
    an array
    """

    # Display the image and plot all contours found
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()

    # generate plot for lithic characterization output
    fig, ax = plt.subplots(figsize=(20, 12))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)

    # sort contours of lithic shape and flake scar outlines by size(area).
    contours.sort_values(by=["area_px"], inplace=True, ascending=False)
    surfaces_classification = utils.classify_surfaces(contours)

    id = 0
    for contour, parent_index, index, area_mm, width_mm, height_mm, angle, polygon_count in \
            contours[['contour', 'parent_index', 'index', 'area_px',
                      'width_mm', 'height_mm', 'angle', 'polygon_count']].itertuples(index=False):
        try:
            if parent_index == -1:
                line_width = 3
                line_style = 'solid'
                classification = surfaces_classification[id]
                text = str(classification) + \
                    ", index: "+str(index) + ",surface_id: "+str(id)+", w: "+str(width_mm)+", h: "+str(height_mm)
                id = id + 1
                ax.plot(contour[:, 0], contour[:, 1], linewidth=line_width, linestyle=line_style, label=text)

            else:
                if math.isnan(angle) == False:
                    line_width = 2
                    line_style = 'solid'
                    # text = "arrow angle: "+str(angle)
                    text = "n vertices: "+str(polygon_count)
                    ax.plot(contour[:, 0], contour[:, 1], linewidth=line_width, linestyle=line_style, label=text)
                else:
                    line_width = 2
                    line_style = 'dashed'
                    text = "n vertices: "+str(polygon_count)
                    ax.plot(contour[:, 0], contour[:, 1], linewidth=line_width, linestyle=line_style, label=text)

        except:
            continue

    fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc='lower left', ncol=2, mode="expand", borderaxespad=0., fontsize='xx-small')
        # bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')

    plt.figtext(0.02, 0.5, str(len(contours))+' contours')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path)
    plt.close(fig)


def plot_thresholding(image_array, threshold, binary_array, output_file=''):
    """
    Visualize the effect of the thresholding on images. Produces three plots, l:r1) original,
    2) pixel intensity histogram, 3) thresholded image.

    Parameters
    ----------
    image_array:  array
        array of an unprocessed image read by openCV (0:255 pixels)
    threshold: float
        threshold value found for images
    binary_array: array
        resulting binary image with pixel values of 0,1
    Returns
    -------
    an array
    """

    image_array_nonzero = image_array > 0

    mean = round(np.mean(image_array[image_array_nonzero]), 2)
    std = round(np.std(image_array[image_array_nonzero]), 2)

    # if mean > 0.9 and std < 0.15:
    #     text = 'segmentation'
    # else:
    #     text = 'edge detection'

    text = 'edge detection'
    fig, axes = plt.subplots(ncols=3, figsize=(10, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(image_array, cmap=plt.cm.gray)
    ax[0].set_title('Processed original')
    ax[0].axis('off')

    ax[1].hist(image_array.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(threshold, color='r')
    # ax[1].text(2, 0.65,"mean: "+str(mean))
    # ax[1].text(1, 0.55,"std: "+ str(std))

    ax[2].imshow(binary_array, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded and '+text)
    ax[2].axis('off')

    if output_file != "":
        plt.savefig(output_file)
    plt.close(fig)


def plot_contour_figure(image_array, cont):
    """
    Returns plots of image contours by color. Waiting on plot design from RAF.

    Parameters
    ----------
    image_array: array
        array of an unprocessed image read by openCV (0:255 pixels)
    cont:
        array of coordinates for a contour

    Returns
    -------
    an array
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)
    ax.plot(cont[:, 0], cont[:, 1])
    plt.close(fig)


def plot_arrow_contours(image_array, contours, output_path):
    """
    Plot the result of the object characterisation
    Parameters
    ----------
    image_array: array
        array of an unprocessed image read by openCV (0:255 pixels)
    contours:
    output_path:

    Returns
    -------

    """

    # Display the image and plot all contours found
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)

    contours.sort_values(by=["area_px"], inplace=True, ascending=False)

    for contour, parent_index, index, area_mm, width_mm, height_mm, arrow in contours[['contour', 'parent_index',
                                                                                       'index', 'area_px', 'width_mm',
                                                                                       'height_mm', 'arrow']].itertuples(index=False):
        try:
            if arrow == True:
                line_style = 'solid'
                ax.plot(contour[:, 0], contour[:, 1], line_width=1, line_style=line_style)
            else:
                continue

        except:
            continue

    fontP.set_size('xx-small')
    plt.figtext(0.02, 0.5, str(len(contours))+' contours')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path)
    plt.close(fig)


def plot_template_arrow(image_array, template, value):
    """
    Plot arrows for associated scars.

    Parameters
    ----------
    image_array: array
        array of an unprocessed image read by openCV (0:255 pixels)
    template
    value
    Returns
    -------

    """

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    ax[0].imshow(image_array, cmap=plt.cm.gray)
    ax[1].imshow(template, cmap=plt.cm.gray)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.figtext(0.4, 0.9, str(value))
    plt.show()
