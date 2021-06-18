# black = background, white = foreground
#
import cv2
import argparse


# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to dataset of images")
ap.add_argument("-c", "--connectivity", type=int, default=4,
                help="connectivity for component analysis")
args = vars(ap.parse_args())

# load the image, convert to gray, and threshold
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imshow("Gray", gray)
cv2.imshow("Threshold", thresh)

# apply connected component analysis to the threshold image
output = cv2.connectedComponentsWithStats(
	thresh, args["connectivity"], cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

# loop over the number of unique connected component labels
for i in range(0, numLabels):
    # if this is the first component the we examine the *background*, r black pixels.
    # typically this component is ignored in the loop, however it needs to be set to some value
    if i == 0:
        text = "examining component {}/{} (background)".format(i + 1, numLabels)
    # examine actual connected component.
    else:
        text = "examining component {}/{}".format(i + 1, numLabels)

    # print a status message update for current connected component
    print("[INFO] {}".format(text))

    # extract connected component stats and centroid
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]

    # clone image so it can be drawn on - bounding box surrounding connected components
    # along with circle corresponding to the centroid
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

    # construct a mask for the connected component by finding pixels in the labels array
    # that have the current connected component ID
    componentMask = (labels == i).astype("uint8") * 255
    # show output
    cv2.imshow("Output", output)
    cv2.imshow("Connected Component", componentMask)
    cv2.waitKey(0)
