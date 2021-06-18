# imports:
import cv2
import numpy as np
import math

# import image and grayscale
image = cv2.imread("templates/a_210.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = 255 - image

# Extend the borders for the skeleton
extended = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT)

# Create a copy of the crop for results:
gray_copy = cv2.cvtColor(extended, cv2.COLOR_GRAY2BGR)

# Create skeleton of the image
skeleton = cv2.ximgproc.thinning(extended, None, 1)

# Threshold the image. White pixels = 0, and black pixels = 10:
# ret = value used for thresholding, thresh is the image used for thresholding
retval, thresh = cv2.threshold(skeleton, 128, 10, cv2.THRESH_BINARY)

# Set the end-points kernel for image convolution

end_points = np.array([[1, 1, 1],
                       [1, 9, 1],
                       [1, 1, 1]])

# Convolve the image using the kernel
filtered_image = cv2.filter2D(thresh, -1, end_points)

# Extract only the end-points pixels with pixel intensity value of 110
binary_image = np.where(filtered_image == 110, 255, 0)

# The above operation converted the image to 32-bit float,
# convert back to 8-bit
binary_image = binary_image.astype(np.uint8)

# Find the X, Y location of all the end-points pixels
Y, X = binary_image.nonzero()

# Reshape arrays for K-means
Y = Y.reshape(-1, 1)
X = X.reshape(-1, 1)
Z = np.hstack((X, Y))

# K-means operates on 32-bit float data
float_points = np.float32(Z)

# Set the convergence criteria and call K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(float_points, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Set the cluster count, find the points belonging
# to cluster 0 and cluster 1

cluster_1_count = np.count_nonzero(label)
cluster_0_count = np.shape(label)[0] - cluster_1_count
#
# print(f"Elements of Cluster 0: {cluster_0_count}")
# print(f"Elements of Cluster 1: {cluster_1_count}")

# The cluster of max number of points will be the tip of the arrow
max_cluster = 0
if cluster_1_count > cluster_0_count:
    max_cluster = 1

# Check out the centers of each cluster:
mat_rows, mat_cols = center.shape

# Store the ordered end-points
ordered_points = [None] * 2

# Identify and draw the two end-points of the arrow
for b in range(mat_rows):
    # Find cluster center
    point_X = int(center[b][0])
    point_Y = int(center[b][1])
    # Get the arrow tip
    if b == max_cluster:
        color = (0, 0, 255)
        ordered_points[1] = (point_X, point_Y)
        cv2.circle(gray_copy, (point_X, point_Y), 5, color, -1)
    # Find the tail
    else:
        color = (255, 0, 0)
        ordered_points[0] = (point_X, point_Y)
        cv2.circle(gray_copy, (point_X, point_Y), 5, color, -1)

# Store the tip and tail points
p0x = ordered_points[1][0]
p0y = ordered_points[1][1]
p1x = ordered_points[0][0]
p1y = ordered_points[0][1]

# Create a new image using the input dimensions
image_height, image_width = binary_image.shape[:2]
new_image = np.zeros((image_height, image_width), np.uint8)
detected_line = 255 - new_image

# Draw a line using the detected points
(x1, y1) = ordered_points[0]
(x2, y2) = ordered_points[1]
cv2.line(detected_line, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)


# Compute x/y distance
(dx, dy) = (p1x-p0x, p1y-p0y)
rads = math.atan2(-dy,dx)
rads %= 2*math.pi
angle_1 = math.degrees(rads)
print(f"Arrow angle: {360-angle_1+90}")

# Compute the angle
angle_2 = math.atan(float(dx)/float(dy))
# The angle is now in radians (-pi/2 to +pi/2).
# change to degrees
angle_2 *= 180/math.pi
print(f"Hypotenuse/opposite angle: {angle_2*-1}")
# Angle is from from -90 to +90
# To flip below the line
if dy < 0:
   angle_2 += 180
print(f"Obtuse angle: {angle_2}")

# angle derived from endpoints of the line drawn along the arrow
line_angle = (180/math.pi)*math.atan(y2/y1)
print(print(f"Line angle: {line_angle}"))

# Show it all
cv2.imshow("End-Points", gray_copy)
# cv2.imshow("Extend borders", extended)
# cv2.imshow("Threshold", thresh)
cv2.imshow("Skeleton", skeleton)
# cv2.imshow("Gray", gray)
cv2.imshow("Binary", binary_image)
cv2.imshow("Detected Line", detected_line)

cv2.waitKey(0)
