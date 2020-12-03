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
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path)
