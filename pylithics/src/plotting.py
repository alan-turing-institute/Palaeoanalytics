import matplotlib.pyplot as plt
import numpy as np

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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)

    for contour, parent_index, index, area_mm, width_mm, height_mm  in contours[['contour', 'parent_index', 'index','area_mm','width_mm','height_mm']].itertuples(index=False):
        try:
            if parent_index==-1:
                linewidth = 3
                linestyle = 'solid'
                text = "L, index: "+str(index)+ ", a: "+str(area_mm)+", w: "+str(height_mm)+", h: "+str(width_mm)
            else:
                linewidth = 2
                linestyle = 'dashed'
                text = "S, p_index: "+str(parent_index)


            ax.plot(contour[:, 0], contour[:, 1], linewidth=linewidth, linestyle=linestyle, label=text)
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

    if mean > 0.9 and std < 0.15:
        text = 'segmentation'
    else:
        text = 'edge detection'

    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(image_array, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(image_array.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(threshold, color='r')
    ax[1].text(2, 0.65,"mean: "+str(mean))
    ax[1].text(1, 0.55,"std: "+ str(std))

    ax[2].imshow(binary_array, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded and '+text)
    ax[2].axis('off')

    if output_file!="":
        plt.savefig(output_file)
    plt.close(fig)
