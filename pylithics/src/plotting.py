import matplotlib.pyplot as plt
import numpy as np
import math
from pylithics.src.utils import classify_surfaces

def plot_contours(image_array, contours, output_path):
    """

    Plot the result of the object characterisation

    Parameters
    ----------
    image_array: array
        Image
    contours
    output_path

    Returns
    -------

    """
    # Display the image and plot all contours found
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()

    fig, ax = plt.subplots(figsize=(16, 10))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)

    contours.sort_values(by=["area_px"], inplace = True, ascending=False)
    surfaces_classification = classify_surfaces(contours)


    id = 0
    for contour, parent_index, index, area_mm, width_mm, height_mm, arrow, angle in contours[['contour', 'parent_index', 'index','area_px','width_mm','height_mm', 'arrow_template_id','angle']].itertuples(index=False):
        try:
            if parent_index==-1:
                linewidth = 3
                linestyle = 'solid'
                classification = surfaces_classification[id]
                text = str(classification)+", index: "+str(index)+ ", surface_id: "+str(id)+", w: "+str(width_mm)+", h: "+str(height_mm)
                id = id + 1
                ax.plot(contour[:, 0], contour[:, 1], linewidth=linewidth, linestyle=linestyle, label=text)

            else:
                if math.isnan(angle)==False:
                    linewidth = 2
                    linestyle = 'solid'
                    text = "arrow angle: "+str(angle)
                    ax.plot(contour[:, 0], contour[:, 1], linewidth=linewidth, linestyle=linestyle, label=text)
                else:
                    linewidth = 2
                    linestyle = 'dashed'
                    ax.plot(contour[:, 0], contour[:, 1], linewidth=linewidth, linestyle=linestyle)

        except:
            continue

    fontP.set_size('xx-small')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')

    plt.figtext(0.02, 0.5, str(len(contours))+' contours')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path)
    plt.close(fig)


def plot_thresholding(image_array, threshold, binary_array, output_file=''):
    """
    Looking at the effect of the thresholding on the image

    Parameters
    ----------
    image_array:  array
        Image
    threshold: float
        threshold value found
    binary_array: array
        resulting binary image

    Returns
    -------

    """

    image_array_nonzero = image_array > 0

    mean = round(np.mean(image_array[image_array_nonzero]),2)
    std = round(np.std(image_array[image_array_nonzero]),2)

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
 #   ax[1].text(2, 0.65,"mean: "+str(mean))
 #   ax[1].text(1, 0.55,"std: "+ str(std))

    ax[2].imshow(binary_array, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded and '+text)
    ax[2].axis('off')

    if output_file!="":
        plt.savefig(output_file)
    plt.close(fig)


def plot_contour_figure(image_array, cont):
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
        Image
    contours
    output_path

    Returns
    -------

    """
    # Display the image and plot all contours found
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)

    contours.sort_values(by=["area_px"], inplace = True, ascending=False)

    for contour, parent_index, index, area_mm, width_mm, height_mm, arrow in contours[['contour', 'parent_index', 'index','area_px','width_mm','height_mm', 'arrow']].itertuples(index=False):
        try:
            if arrow == True:
                linestyle = 'solid'
                ax.plot(contour[:, 0], contour[:, 1], linewidth=1, linestyle=linestyle)
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


