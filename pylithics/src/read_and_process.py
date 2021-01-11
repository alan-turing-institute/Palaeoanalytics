import skimage
import skimage.feature
import skimage.viewer
from skimage.filters import threshold_minimum, threshold_mean
from skimage import filters
from skimage.measure import find_contours
import numpy as np
from skimage.filters.rank import median

from skimage import exposure

import matplotlib.pyplot as plt

def read_image(filename):
    """
    Function that read an image into the skimage library

    Parameters
    ==========
    filename: str, path and file name to the directory where the image
    Returns
    =======
    an array
    """
    image = skimage.io.imread(fname=filename, as_gray=True)

    return image

def detect_lithic(image_array, config_file):
    """
    Function that given an input image array and configuration options
    applies thresholding
    and edge detection to find the general shape of the lithic object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array
    a float with the threshold value

    """

    thresh = threshold_minimum(image_array)
    thresh = thresh*config_file['threshold']

    binary = image_array < thresh

    binary_edge_sobel = filters.sobel(binary)

    return binary_edge_sobel, thresh

def find_lithic_contours(image_array, config_file):
    """
    Function that given an input image array and configuration options
     finds contours on a the lithic object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array

    """

    contours = find_contours(image_array, config_file['contour_parameter'], fully_connected=config_file['contour_fully_connected'])

    new_contours = []
    for cont in contours:
        if cont.shape[0] < config_file['minimum_pixels_contour']:
            continue
        else:
            new_contours.append(cont)

    new_contours = np.array(new_contours, dtype="object")

    return new_contours

def detect_scale(image_array, config_file):
    """
    Function that given an input image array and configuration options
    applies thresholding
    and edge detection to find the scale for the lithic object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array

    """

    thresh = threshold_mean(image_array)
    thresh = thresh*config_file['threshold']

    binary = image_array < thresh

    binary_edge_sobel = filters.sobel_h(binary)

    return binary_edge_sobel, thresh

def find_scale_contours(image_array, config_file):
    """
    Function that given an input image array and configuration options
     finds contours on the scale object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array

    """

    contours = find_contours(image_array, config_file['contour_parameter'], fully_connected=config_file['contour_fully_connected'])

    lrg_contour = sorted(contours, key = len)[-1]
    return lrg_contour


def detect_lithic_(image_array, config_file):
    """
    Function that given an input image array and configuration options
    applies thresholding
    and edge detection to find the general shape of the lithic object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array
    a float with the threshold value

    """


    # Load an example image

    img_denoise = denoise(image_array)
    img_eq = equialisation(img_denoise)

    image_array = img_eq
    thresh = threshold_mean(image_array)
    thresh = thresh+thresh*config_file['threshold']

    binary = image_array < thresh

    binary_edge_sobel = filters.sobel(binary)

    return binary_edge_sobel, thresh, image_array

def equialisation(img):


    import matplotlib
    import matplotlib.pyplot as plt

    from skimage import data, img_as_float
    from skimage import exposure

    matplotlib.rcParams['font.size'] = 8

    def plot_img_and_hist(image, axes, bins=256):
        """Plot an image along with its histogram and cumulative histogram.

        """
        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf

    # Contrast stretching
    p2, p98 = np.percentile(img, (4, 96))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    try:
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.005)
    except:
        img_adapteq = img

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 4), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title('Contrast stretching')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title('Histogram equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title('Adaptive equalization')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    #plt.show()

    return img_rescale

def denoise(noisy):

    import matplotlib.pyplot as plt

    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                     denoise_wavelet, estimate_sigma)
    from skimage import data, img_as_float
    from skimage.util import random_noise
    from skimage.morphology import disk, ball

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 5),
                           sharex=True, sharey=True)

    plt.gray()

    # Estimate the average noise standard deviation across color channels.
    sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
    # Due to clipping in random_noise, the estimate will be a bit smaller than the
    # specified sigma.
    print(f"Estimated Gaussian noise standard deviation = {sigma_est}")

    ax[0].imshow(noisy)
    ax[0].axis('off')
    ax[0].set_title('Noisy')
    ax[1].imshow(denoise_tv_chambolle(noisy, weight=0.03, multichannel=False))
    ax[1].axis('off')
    ax[1].set_title('TV')
    ax[2].imshow(denoise_bilateral(noisy, sigma_spatial=5,
                                      multichannel=False))
    ax[2].axis('off')
    ax[2].set_title('Bilateral')
    ax[3].imshow(denoise_wavelet(noisy, multichannel=False, rescale_sigma=True))
    ax[3].axis('off')
    ax[3].set_title('Wavelet denoising')

    ax[4].imshow(median(noisy, disk(1)))
    ax[4].axis('off')
    ax[4].set_title('Median')

    fig.tight_layout()

    #plt.show()

    array = denoise_tv_chambolle(noisy, weight=0.05, multichannel=False)

    return array