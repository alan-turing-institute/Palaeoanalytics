import matplotlib.pyplot as plt
import numpy as np
import math
import pylithics.src.utils as utils
# Display the image and plot all contours found
from matplotlib.font_manager import FontProperties
import os
import matplotlib as mpl

def fig_size(image_array):
    """

    Calculate optimum figure size for a plot based on the ratio of sizes of rows and columns.

    Parameters
    ----------
    image_array: array
    Original image array (0 to 255)

    Returns
    -------

    A number for the x width of the figure size.

    """
    ratio = image_array.shape[0] / image_array.shape[1]
    if ratio < 1:
        fig_size = 20 / ratio
    else:
        fig_size = 10 * ratio

    return fig_size

def plot_surfaces(image_array, contours_df, output_path):
    """
    Plot the contours from the lithic surfaces.

    Parameters
    ----------
    image_array: array
    Original image array (0 to 255)
    contours_df:
        Dataframe with detected contours and extra information about them.
    output_path:
        path to output directory to save processed images

    """
    fig_x_size = fig_size(image_array)
    fig, ax = plt.subplots(figsize=(fig_x_size, 20))
    ax.imshow(image_array, cmap=plt.cm.gray)

    surfaces_classification = utils.classify_surfaces(contours_df)
    # selecting only surfaces (lowest hiearchy level).
    contours_surface_df = contours_df[contours_df['parent_index'] == -1].sort_values(by=["area_px"], ascending=False)

    i = 0
    for contour in contours_surface_df['contour'].values:
        classification = surfaces_classification[i]
        text = str(classification)
        ax.plot(contour[:, 0], contour[:, 1], label=text, linewidth=5)
        i = i + 1

    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=17)
    plt.title("Detected surfaces", fontsize=25)
    plt.show()
    plt.savefig(output_path)
    plt.close(fig)


def plot_scars(image_array, contours_df, output_path):
    """
    Plot the contours from the lithic surfaces.

    Parameters
    ----------
    image_array: array
    Original image array (0 to 255)
    contours_df:
        Dataframe with detected contours and extra information about them.
    output_path:
        path to output directory to save processed images

    """
    fig_x_size = fig_size(image_array)
    fig, ax = plt.subplots(figsize=(fig_x_size, 20))

    ax.imshow(image_array, cmap=plt.cm.gray)

    # selecting only surfaces (lowest hiearchy level).
    contours_surface_df = contours_df[contours_df['parent_index'] != -1].sort_values(by=["area_px"], ascending=False)

    i = 0
    for contour, area_mm, width_mm, height_mm in \
            contours_surface_df[['contour', 'area_mm',
                                 'width_mm', 'height_mm']].itertuples(index=False):
        text = "A: " + str(area_mm) + ", B: " + str(width_mm) + ", L: " + str(height_mm)
        ax.plot(contour[:, 0], contour[:, 1], label=text, linewidth=4)
        i = i + 1

    ax.set_xticks([])
    ax.set_yticks([])
    plt.figtext(0.05, 0.5, ("A: Total Area"), fontsize=20)
    plt.figtext(0.05, 0.52, ("B: Maximum Breath"), fontsize=20)
    plt.figtext(0.05, 0.54, ("L: Maximum Lenght"), fontsize=20)
    plt.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=11)
    plt.title("Scar measurements (in millimeters)", fontsize=25)
    plt.show()
    plt.savefig(output_path)
    plt.close(fig)


def plot_results(id, image_array, contours_df, output_dir):
    """
    Plot the results of the object characterisation.

    Parameters
    ----------
    id: str
        Name of the lithic
    image_array: array
         Original image array (0 to 255)
    contours_df:

    output_path:
        path to output directory to save processed images

        Returns
        -------
        an array
    """

    # plot surfaces
    output_lithic = os.path.join(output_dir, id + "_lithic_surfaces.png")
    plot_surfaces(image_array, contours_df, output_lithic)

    output_lithic = os.path.join(output_dir, id + "_lithium_scars.png")
    plot_scars(image_array, contours_df, output_lithic)


def plot_contours(image_array, contours, output_path):
    """
    Plot the results of the object characterisation.

    Parameters
    ----------
    image_array: array
    Original image array (0 to 255)
    contours:

    output_path:
        path to output directory to save processed images

    Returns
    -------
    an array
    """

    # Display the image and plot all contours found
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()

    fig, ax = plt.subplots(figsize=(20, 12))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)

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
                       ", index: " + str(index) + ",surface_id: " + str(id) + ", w: " + str(width_mm) + ", h: " + str(
                    height_mm)
                id = id + 1
                ax.plot(contour[:, 0], contour[:, 1], line_width=line_width, line_style=line_style, label=text)

            else:
                if math.isnan(angle) == False:
                    line_width = 2
                    line_style = 'solid'
                    # text = "arrow angle: "+str(angle)
                    text = "n vertices: " + str(polygon_count)
                    ax.plot(contour[:, 0], contour[:, 1], line_width=line_width, line_style=line_style, label=text)
                else:
                    line_width = 2
                    line_style = 'dashed'
                    text = "n vertices: " + str(polygon_count)
                    ax.plot(contour[:, 0], contour[:, 1], line_width=line_width, line_style=line_style, label=text)

        except:
            continue

    fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc='lower left', ncol=2, mode="expand", borderaxespad=0., fontsize='xx-small')
    # bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')

    plt.figtext(0.02, 0.5, str(len(contours)) + ' contours')
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
        Original image array (0 to 255)
    threshold: float
        threshold value found for images
    binary_array: array
        resulting binary image with pixel values of... <- variable name

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
    ax[2].set_title('Thresholded and ' + text)
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

    cont:

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
                                                                                       'height_mm',
                                                                                       'arrow']].itertuples(
        index=False):
        try:
            if arrow == True:
                line_style = 'solid'
                ax.plot(contour[:, 0], contour[:, 1], line_width=1, line_style=line_style)
            else:
                continue

        except:
            continue

    fontP.set_size('xx-small')
    plt.figtext(0.02, 0.5, str(len(contours)) + ' contours')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path)
    plt.close(fig)


def plot_template_arrow(image_array, template, value):
    """
    Plot arrows for associated scars.

    Parameters
    ----------
    image_array
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
