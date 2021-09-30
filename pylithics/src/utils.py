import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import pylithics.src.plotting as plot
import math
from scipy.spatial.distance import cdist


def mask_image(binary_array, contour, innermask=False):
    """
    Mask negative (white pixels) areas of lithic flakes scars in order to generate contour lines.

    Parameters
    ----------
    binary_array: array
        Array of a processed image  (0, 1 pixels)
    contour: list of lists
        Pixel coordinates for a contour of a single flake scar
        or outline of a lithic object detected by contour finding

    Returns
    -------
    An image array
    """

    r_mask = np.zeros_like(binary_array, dtype='bool')
    r_mask[np.round(contour[:, 1]).astype('int'), np.round(contour[:, 0]).astype('int')] = 1

    r_mask = ndi.binary_fill_holes(r_mask)

    if innermask:
        new_image = r_mask
    else:
        new_image = np.multiply(r_mask, binary_array)

    return new_image


def contour_characterisation(image_array, contour, conversion=1):
    """
    Calculate contour characteristics (area, length, etc.),
    from a specific element of an image.

    Parameters
    ----------
    contour: list of lists
        Pixel coordinates for a contour of a single flake scar
        or outline of a lithic object detected by contour finding
    conversion: float
        Value to convert pixels to millimeters

    Returns
    -------
    A dictionary

    """
    contour_info = {}

    # Expand numpy dimensions and convert it to UMat object
    area = cv2.contourArea(contour)

    masked_image = mask_image(image_array, contour, True)

    contour_info['length'] = len(contour * conversion)
    contour_info['area_px'] = area

    rows, columns = subtract_masked_image(masked_image)

    # height and width of individual flake scar/lithic surface outline in pixels
    contour_info['height_px'] = masked_image.shape[0] - len(rows)
    contour_info['width_px'] = masked_image.shape[1] - len(columns)

    contour_info['centroid'] = ndi.center_of_mass(mask_image(image_array, contour, True))

    if conversion == 1:
        area_mm = np.nan
        width_mm = np.nan
        height_mm = np.nan
    else:
        area_mm = round(area * (conversion * conversion), 1)
        width_mm = round(contour_info['width_px'] * conversion, 1)
        height_mm = round(contour_info['height_px'] * conversion, 1)

    contour_info['area_mm'] = area_mm
    contour_info['width_mm'] = width_mm
    contour_info['height_mm'] = height_mm
    contour_info['polygon_count'], _ = measure_vertices(contour, 0.02)

    return contour_info


def classify_distributions(image_array):
    """
    Classifies an image array by its distribution of pixel intensities.
    Returns True if the distribution is narrow and skewed to values of 1.

    Parameters
    ----------
    image_array: array
        Array of an unprocessed image  (0,255 pixels)
    Returns
    -------
    a boolean
    """

    is_narrow = False

    image_array_nonzero = image_array > 0

    mean = np.mean(image_array[image_array_nonzero])

    std = np.std(image_array[image_array_nonzero])

    if mean > 0.9 and std < 0.15:
        is_narrow = True

    return is_narrow


def get_high_level_parent_and_hierarchy(hierarchies):
    """
    Given a list of individual contour hierarchies, find the index of the highest level parent for each contour.
    Defines which scar belongs to which surface.

     Parameters
    ----------
    hierarchies: list
        A list of contour hierarchies

    Returns
    -------
    A list
    """

    parent_index = []
    hierarchy_level = []

    for index, hierarchy in enumerate(hierarchies, start=0):

        parent = hierarchy[-1]
        count = 0

        if parent == -1:
            parent_index.append(parent)
            hierarchy_level.append(count)
        else:
            while (parent != -1):
                index = parent
                parent = hierarchies[index][-1]
                count = count + 1

            parent_index.append(index)
            hierarchy_level.append(count)

    return parent_index, hierarchy_level


def pixulator(image_scale_array, scale_size):
    """
    Converts image and scale pixel count to millimeters.

    Parameters
    ----------
    image_scale_array: Array
        Image array of scale data.
    scale_size:
        Length of the scale in mm

    Returns
    -------
    Conversion image pixels to millimeters.
    """

    # dimension information in pixels
    px_width = image_scale_array.shape[0]
    px_height = image_scale_array.shape[1]

    if px_width > px_height:
        orientation = px_width
    else:
        orientation = px_height

    px_conversion = 1 / (orientation / scale_size)


    print(f"1 cm will equate to {round(1 / px_conversion,1)} pixels width.")

    return (px_conversion)


def classify_surfaces(contour_df):
    """
    Classify individual surfaces into Ventral, Dorsal, Lateral or Platform

    Parameters
    ----------
    contour_df: dataframe
        Dataframe with all the contour information and measurements for an image

    Returns
    -------
    A dataframe
    """

    def dorsal_ventral(contour_df, surfaces_df):
        """

        Parameters
        ----------
        contour_df: dataframe
            Dataframe with all the contour information and measurements for an image
        surfaces_df: dataframe
            Dataframe with all the contour information and measurements related to the surfaces of an image

        Returns
        -------
        A dictionary with a surface classification

        """

        output = [None] * 2
        if (contour_df[contour_df['parent_index'] == surfaces_df['index'].iloc[0]].shape[0] >
                contour_df[contour_df['parent_index'] == surfaces_df['index'].iloc[1]].shape[0]):

            output[0] = 'Dorsal'
            output[1] = 'Ventral'
        else:
            output[0] = 'Ventral'
            output[1] = 'Dorsal'

        return output


    # dataframe should be sorted in order for this algorithm to work correctly.
    surfaces = contour_df[contour_df['hierarchy_level'] == 0].sort_values(by=["area_px"], ascending=False)  #

    names = {}
    # start assigning them all to nan
    for i in range(surfaces.shape[0]):
        names[i] = 'Unclassified'

    # Dorsal, lateral, platform, ventral.
    if surfaces.shape[0] == 1:
        names[0] = 'Dorsal'

    elif surfaces.shape[0] > 1:
        ratio = surfaces["area_px"].iloc[1] / surfaces["area_px"].iloc[0]

        if ratio > 0.9:
            names[0], names[1] = dorsal_ventral(contour_df, surfaces)

        elif surfaces.shape[0] == 2 and ratio <= 0.9:

            if ratio > 0.3:
                names[0] = 'Dorsal'
                names[1] = 'Lateral'
            else:
                names[0] = 'Dorsal'
                names[1] = 'Platform'

        if surfaces.shape[0] == 3:
            if ratio > 0.9:

                ratio2 = surfaces["area_px"].iloc[2] / surfaces["area_px"].iloc[0]

                if ratio2 > 0.2:
                    names[2] = 'Lateral'
                else:
                    names[2] = 'Platform'
            else:
                names[0] = 'Dorsal'
                names[1] = 'Lateral'
                names[2] = 'Platform'


        elif surfaces.shape[0] > 3:
            names[0], names[1] = dorsal_ventral(contour_df, surfaces)
            ratio2 = surfaces["area_px"].iloc[2] / surfaces["area_px"].iloc[0]
            if ratio2 > 0.2:
                names[2] = 'Lateral'
            else:
                names[2] = 'Platform'

            if surfaces["area_px"].iloc[3] / surfaces["area_px"].iloc[0] < 0.2:
                names[3] = 'Platform'

    return names


def subtract_masked_image(masked_image_array):
    """
    Given an input masked image, return all rows and columns where an image is all masked.

    Parameters
    ----------
    masked_image_array: array
        A masked image array

    Returns
    -------
    A list
    """

    # Check rows in which all values are equal
    rows = []
    for i in range(masked_image_array.shape[0]):
        if np.all(masked_image_array[i] == False):
            rows.append(i)
    # Check Columns in which all values are equal
    columns = []
    trans_masked_image = masked_image_array.T
    for i in range(trans_masked_image.shape[0]):
        if np.all(trans_masked_image[i] == False):
            columns.append(i)

    return rows, columns


def template_matching(image_array, templates_df, contour, debug=False):
    """

    Find best template match in an image within a contour

    Parameters
    ----------
    image_array: array
        Array of input masked_image_array
    templates_df: array
        Array of template images
    contour:
        Pixel coordinates for a contour
    debug: bool
        Generate plot if True

    Returns
    -------
    Index of best matching template
    """

    # get template array from dataframe
    templates = templates_df['template_array'].values

    # reduce image to only the scar
    rows, columns = subtract_masked_image(image_array)
    masked_image = np.delete(image_array, rows[:-1], 0)
    masked_image = np.delete(masked_image, columns[:-1], 1)

    # this is to be able to do things with cv2
    image_array = image_array.astype(np.float32)

    # default values of the index of best matched template and
    location_index = -1
    avg_match = 0

    # go through each template and find the best matching one.
    for i, template in enumerate(templates):
        (tW, tH) = template.shape[::-1]
        (sW, sH) = image_array.shape[::-1]
        if tW > sW or tH > sH:
            continue

        # template matching
        result = cv2.matchTemplate(image_array, template.astype(np.float32), cv2.TM_CCORR_NORMED)

        # areas where results are >= than threshold value
        location = np.where((result >= 0.93) & (result < 1.0))

        location_new_x = []
        location_new_y = []

        # make sure the matched template is inside the scar contours
        for j in range(len(location[0])):
            X = location[1][j]
            Y = location[0][j]
            inside = cv2.pointPolygonTest(contour, (int(X), int(Y)), False)
            if inside == 1.0:
                location_new_x.append(location[1][j])
                location_new_y.append(location[0][j])
        location_new = (np.array(location_new_y), np.array(location_new_x))

        # save template index is the average result values is best than previous one
        if len(location_new[0]) > 0:
            if result[location_new].mean() > avg_match:
                avg_match = result[location_new].mean()
                location_index = i

    # plot the matching scar and arrow
    if location_index != -1 and debug == True:
        plot.plot_template_arrow(masked_image, templates[location_index], avg_match)

    return location_index


def get_angles(templates):
    """
    Create a dataframe of angles measured from arrows.

    Parameters
    ----------
    templates: list
        A list of template image arrays
    Returns
    -------
    A dataframe

    """

    template_dict_list = []

    for index, template in enumerate(templates):

        template_dict = {}

        template_dict['template_array'] = template
        try:
            template_dict['angle'] = measure_arrow_angle(template)
        except:
            template_dict['angle'] = np.nan

        template_dict_list.append(template_dict)

    templates_df = pd.DataFrame.from_records(template_dict_list)

    return templates_df


def contour_selection(contour_df):

    """
    Function that selects contours by their size and removes duplicates.

    Parameters
    ----------
    contour_df: dataframe
        Dataframe with all contour information for an image.

    Returns
    -------
    A list of indexes

    """

    index_to_drop = []

    for hierarchy_level, index, parent_index, area_px, contour, in contour_df[
        ['hierarchy_level', 'index', 'parent_index', 'area_px', 'contour']].itertuples(index=False):

        pass_selection = True
        if hierarchy_level == 0:
            if area_px / max(contour_df['area_px']) < 0.03:
                pass_selection = False

        else:
            # only allow low hierarchies
            if hierarchy_level > 2:
                pass_selection = False
            else:
                # get the total area of the parent figure
                norm = contour_df[contour_df['index'] == parent_index]['area_px'].values[0]
                area = area_px
                percentage = area / norm
                if percentage < 0.01:
                    pass_selection = False
                if percentage > 0.50:
                    pass_selection = False

        if not pass_selection:
            index_to_drop.append(index)

    return index_to_drop


def measure_arrow_angle(template):
    """
    Function that measures the angle of an arrow template.

    Parameters
    ----------
    template: array
        2d array representing an arrow.

    Returns
    -------
    An angle measurement
    """

    # import image and grayscale
    uint_img = np.array(template * 255).astype('uint8')
    gray = 255 - uint_img

    # Extend the borders for the skeleton for convolution
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
    (dx, dy) = (p0y - p1y, p0x - p1x)  # delta(d) x and delta y (distance between points along
    # x and y axis). Here they have been reversed as angles do not extend from the center
    # but towards it.
    rads = math.atan2(-dy, dx)  # convert to radians
    rads %= 2 * math.pi
    angle = math.degrees(rads)  # convert to degrees.

    return round(angle, 2)


def measure_vertices(contour, epsilon=0.04):
    """

    Given a contour from a surface or scar, estimate the number of vertices of an approximate
    shape to the contour.

    Parameters
    ----------
    contour:
        Pixel coordinates for a contour of a single flake scar
        or outline of a lithic object detected by contour finding.
    epsilon: float
        Degree of prediction in approximation

    Returns
    -------
    A number
    An array with approximate contour
    """

    # get perimeters
    peri = cv2.arcLength(contour, True)
    # approximates a curve or a polygon with another curve / polygon with less vertices
    # so that the distance between them is less or equal to the specified precision
    approx = cv2.approxPolyDP(contour, epsilon * peri, True)

    return len(approx), approx


def shape_detection(contour):
    """
    Given a contour from a surface or scar, detect an approximate shape to the contour.

    Parameters
    ----------
    contour:
        Pixel coordinates for a contour of a single flake scar
        or outline of a lithic object detected by contour finding

    Returns
    -------
    A string with the shape classification
    A number with the number of vertices
    """

    # initialize the shape name and approximate the contour
    shape = "unidentified"
    vertices, approx = measure_vertices(contour)
    # if the shape is a triangle, it will have 3 vertices
    if vertices == 3:
        shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
    elif vertices == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "arrow"
        # if the shape is a pentagon, it will have 5 vertices
    elif vertices >= 5:
        shape = "arrow"
        # otherwise, we assume the shape is an arrow
    return shape, vertices

def complexity_estimator(contour_df):
    """

    Function that estimate a complexity measure. Complexity is measured as the number of adjacent contours
    for each contour.

    Parameters
    ----------
    contour_df: dataframe
        Dataframe with all contour information for an image.
    Returns
    -------

    A copy of the contour_df dataframe with a new measure of complexity

    """

    adjacency_list = []
    for i in range(0, contour_df.shape[0]):
        if contour_df.iloc[i]["parent_index"] == -1:
            adjacency_list.append(0)
        else:
            # list coordinates for the contour we are interested on
            contour_coordinate = contour_df.iloc[i]["contour"]

            # list of coordinates of each of the siblings that we are interested on (list of list)
            contour_coordinate_siblings = contour_df[contour_df["parent_index"] == contour_df.iloc[i]["parent_index"]]['contour'].values

            count = 0
            for sibling_contour in contour_coordinate_siblings:

                # compare contour_coordinate with sibling_contour
                adjacent = complexity_measure(contour_coordinate,sibling_contour)

                if adjacent == True:
                    count = count + 1

            adjacency_list.append(count)

    contour_df['complexity'] = adjacency_list

    return contour_df


def complexity_measure(contour_coordinates1, contour_coordinates2):
    """
    Decide if two contours are adjacent based on distance between its coordinates.

    Parameters
    ----------
    contour_coordinates1: list of lists
        Pixel coordinates for a contour of a single flake scar
        or outline of a lithic object detected by contour finding
    contour_coordinates2: list of lists
        Pixel coordinates for a contour of a single flake scar
        or outline of a lithic object detected by contour finding

    Returns
    -------

    A boolean

    """

    # if they are the same contour they are not adjacent
    if np.array_equal(contour_coordinates1, contour_coordinates2):
        return False
    else:
        # get minimum distance between contours
        min_dist = np.min(cdist(contour_coordinates1, contour_coordinates2))

        # if the minimum distance found is less than a threshold then they are adjacent
        if min_dist < 50:
            return True
        else:
            return False











