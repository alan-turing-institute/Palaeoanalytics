
# convolution = element-wise multiplication of two matrices followed by a sum.

# import packages
import cv2
import numpy as np
from skimage.exposure import rescale_intensity

# what's going on
# step 1: take two matrices with the same dimensions
# step 2: multiply them, element by element (i.e., NOT dot-product, just simple multiplication)
# step 3: sum the elements

# example of elementwise multiplication of matrices
a = np.arange(0, 9).reshape(3, 3)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

b = np.arange(0, 9).reshape(3, 3)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

# a * b <- multiply the matrices by element
# [[ 0  1  4]
#  [ 9 16 25]
#  [36 49 64]]

# sum the output of the two multiplied matrices
# (a * b).sum()
# = 204


def kustom_kernel(image, kernel):
    # get spatial dimensions of...
    (iH, iW) = image.shape[:2]  # the image
    (kH, kW) = kernel.shape[:2]  # the kernel

    # allocate memory for the output image and padded borders so that image spatial dimension are not reduced
    padding = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                               cv2.BORDER_REPLICATE)
    # create output array with the same dimensions of the image
    output = np.zeros((iH, iW), dtype="float32")

    #  loop/slide the kernel across image coordinates left to right, top to bottom
    for y in np.arange(pad, iH + pad):  # rows
        for x in np.arange(pad, iW + pad):  # columns
            # extract ROI of image by extracting the center region of the current (x, y) coordinate dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]  # ROI has the same size as the kernel

            # perform convolution between the ROI and the kernel
            k = (roi * kernel).sum()

            # store the convolved values in the output (x, y) coordinates of the output image
            output[y - pad, x - pad] = k
    #  rescale the output image to be in the range of (0 - 255)
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")  # convert back to unsigned 8-bit integer.

    # output image
    return output