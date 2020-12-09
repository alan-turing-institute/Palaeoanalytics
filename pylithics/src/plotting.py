import matplotlib.pyplot as plt
import os


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
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap=plt.cm.gray)

    for contour in contours:
        try:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        except:
            continue

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path)

def plot_thresholding(image_array, threshold, binary_array, output_file):
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

    ax[2].imshow(binary_array, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')

    plt.savefig(output_file)
