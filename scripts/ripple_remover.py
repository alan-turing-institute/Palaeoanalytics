# morphological operations are generally applied to thresholded or binary images,
# and are used to 'clean up' images before further processing.

import cv2
import argparse

# construct the parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# # resize image to have a width of 800 pixels
# r = 800.0 / image.shape[0]
# dim = (int(image.shape[1] * r), 800)  # now 800 is the new Y while int(image.shape[1] * r) is the new x
# image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # erosion require grayscale/binary images.
gray = cv2.blur(gray, (3, 3), 0)

# otsu thresholding.
(T, threshInv) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", threshInv)
print("[INFO] Otsu's thresholding value: {}".format(T))

# apply a series of three erosions
for i in range(0, 3):
    eroded = cv2.erode(threshInv, None, iterations=i + 1)
    cv2.imshow("Eroded {} times".format(i + 1), eroded)
    cv2.waitKey(0)

# close all windows and clear the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# apply a series of three dilations
for i in range(0, 3):
    eroded = cv2.dilate(threshInv, None, iterations=i + 1)
    cv2.imshow("Dilated {} times".format(i + 1), eroded)
    cv2.waitKey(0)


#####
# opening - an erosion followed by a dilation - cv2.MORPH_OPEN
#####

# close all windows and clear the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)
kernelSizes = [(3, 3), (5, 5), (7, 7)]

# loop over kernel sizes
for kernelSize in kernelSizes:
    # construct a rectangular kernel from the current size and apply an "opening" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    opening = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Opening: ({}, {})".format(
        kernelSize[0], kernelSize[1]), opening)
    cv2.waitKey(0)

#####
# closing - a dilation followed by an erosion = cv2.MORPH_CLOSE
#####

# close all windows and clear the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# loop over kernel sizes
for kernelSize in kernelSizes:
    # construct a rectangular kernel from the current size and apply a "closing" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    closing = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing: ({}, {})".format(
        kernelSize[0], kernelSize[1]), closing)
    cv2.waitKey(0)

#####
# morphological gradient - the difference between a dilation and an erosion = cv2.MORPH_GRADIENT
#####

# close all windows and clear the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# loop over kernel sizes
for kernelSize in kernelSizes:
    # construct a rectangular kernel from the current size and apply a "morphological gradient" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    gradient = cv2.morphologyEx(threshInv, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("Morphological gradient: ({}, {})".format(
        kernelSize[0], kernelSize[1]), gradient)
    cv2.waitKey(0)
