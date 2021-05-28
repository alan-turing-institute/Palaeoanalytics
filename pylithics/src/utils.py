import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from pylithics.src.shapedetector import ShapeDetector


def area_contour(contour):
    """
    Function that calculates the area within cont contour using the open-cv library.

    Parameters
    ----------
    contour: array (array with coordinates defining the contour.)

    Returns
    -------
    A number

    """
    # Expand numpy dimensions and convert it to UMat object
    c = cv2.UMat(np.expand_dims(contour.astype(np.float32), 1))
    area = cv2.contourArea(c)

    return area


def contour_desambiguiation(df_contours, image_array):
    """

    Function that selects contours by their size and removes duplicates.

    Parameters
    ----------
    df_contours: dataframe
        Dataframe with contour information.

    Returns
    -------

    list of indexes

    """



    index_to_drop = []

    for hierarchy_level, index, parent_index, area_px, contour, in df_contours[
        ['hierarchy_level', 'index', 'parent_index', 'area_px', 'contour']].itertuples(index=False):

        pass_selection = True
        if hierarchy_level == 0:
            if area_px / max(df_contours['area_px']) < 0.1:
                pass_selection = False
            else:
                continue
        else:
            # get the total area of the parent figure
            norm = df_contours[df_contours['index'] == parent_index]['area_px'].values[0]
            area = area_px
            percentage = area / norm * 100
            if percentage < 3:
                pass_selection = False
            if percentage > 60:
                pass_selection = False


        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax = plt.subplot(111)
        # ax.imshow(image_array, cmap=plt.cm.gray)
        # ax.plot(contour[:, 0], contour[:, 1])

        if pass_selection == False:
            index_to_drop.append(index)

    cent_df = df_contours[['area_px', 'centroid', 'hierarchy_level','contour']]

    import itertools

    for i, j in itertools.combinations(cent_df.index, 2):

        if ((i in index_to_drop) or (j in index_to_drop)):
            continue


        d_ij_centroid = np.linalg.norm(np.asarray(cent_df.loc[i]['centroid']) - np.asarray(cent_df.loc[j]['centroid']))

        if cent_df.loc[i]['area_px'] > cent_df.loc[j]['area_px']:
            ratio = cent_df.loc[j]['area_px'] / cent_df.loc[i]['area_px']
        else:
            ratio = cent_df.loc[i]['area_px'] / cent_df.loc[j]['area_px']


        if d_ij_centroid < 15 and ratio > 0.5:
            if (cent_df.loc[i]['area_px'] < cent_df.loc[j]['area_px']):
                if d_ij_centroid<1:
                    index_to_drop.append(i)
                else:
                    index_to_drop.append(j)
            elif (cent_df.loc[i]['area_px'] > cent_df.loc[j]['area_px']):
                if d_ij_centroid < 1:
                    index_to_drop.append(j)
                else:
                    index_to_drop.append(i)

    return index_to_drop


def mask_image(image_array, contour, innermask=False):
    """

    Function that masks an image for cont given contour.

    Parameters
    ----------
    image_array
    contour

    Returns
    -------

    """

    r_mask = np.zeros_like(image_array, dtype='bool')
    r_mask[np.round(contour[:, 1]).astype('int'), np.round(contour[:, 0]).astype('int')] = 1

    r_mask = ndi.binary_fill_holes(r_mask)

    if innermask:
        new_image = r_mask
    else:
        new_image = np.multiply(r_mask, image_array)

    return new_image


def contour_characterisation(image_array, cont, conversion=1):
    """

    For cont given contour calculate characteristics (area, lenght, etc.)

    Parameters
    ----------
    cont: array
        Array of pairs of pixel coordinates
    conversion: float
        Value to convert pixels to inches

    Returns
    -------
    A dictionary

    """
    cont_info = {}

    # Expand numpy dimensions and convert it to UMat object
    area = area_contour(cont)

    masked_image = mask_image(image_array, cont, True)

    cont_info['length'] = len(cont * conversion)
    cont_info['area_px'] = area

    rows, columns = subtract_masked_image(masked_image)

    cont_info['height_px'] = masked_image.shape[0] - len(rows)
    cont_info['width_px'] = masked_image.shape[1] - len(columns)

    cont_info['centroid'] = ndi.center_of_mass(mask_image(image_array, cont, True))

    if conversion == 1:
        area_mm = np.nan
        width_mm = np.nan
        height_mm = np.nan
    else:
        area_mm = round(area * (conversion * conversion), 1)
        width_mm = round(cont_info['width_px'] * conversion, 1)
        height_mm = round(cont_info['height_px'] * conversion, 1)

    cont_info['area_mm'] = area_mm
    cont_info['width_mm'] = width_mm
    cont_info['height_mm'] = height_mm
    # cont_info['contour'] = cont

    return cont_info


def classify_distributions(image_array):
    """
    Given an input image array classify it by their distribution of pixel intensities.
    Returns True is the ditribution is narrow and skewed to values of 1.

    Parameters
    ----------
    image_array: array

    Returns
    -------
    a boolean

    """

    is_narrow = False

    fig, axes = plt.subplots(figsize=(8, 2.5))

    axes.hist(image_array.ravel(), bins=256)
    axes.set_title('Histogram')
    plt.close(fig)

    image_array_nonzero = image_array > 0

    mean = np.mean(image_array[image_array_nonzero])

    std = np.std(image_array[image_array_nonzero])

    if mean > 0.9 and std < 0.15:
        is_narrow = True

    return is_narrow


def get_high_level_parent_and_hirarchy(hierarchies):
    """ For a list of contour hierarchies find the index of the
    highest level parent for each contour.

     Parameters
    ----------
    hierarchies: list
        List of hierarchies

    Returns
    -------
    A list
    """

    parent_index = []
    hirarchy_level = []

    for index, hierarchy in enumerate(hierarchies, start=0):

        parent = hierarchy[-1]
        count = 0

        if parent == -1:
            parent_index.append(parent)
            hirarchy_level.append(count)
        else:
            while (parent != -1):
                index = parent
                parent = hierarchies[index][-1]
                count = count + 1

            parent_index.append(index)
            hirarchy_level.append(count)

    return parent_index, hirarchy_level


def pixulator(image_scale_array, scale_size):
    """
    Converts image/scale dpi and pixel count to cm conversion rate.


    Parameters
    ----------
    image_scale_array: array
        Image array
    scale_size:
        length in mm of the scale

    Returns
    -------
        Image conversion pixel to centimeters.
    """

    # dimension information in pixels
    px_width = image_scale_array.shape[0]
    px_height = image_scale_array.shape[1]

    if px_width > px_height:
        orientation = px_width
    else:
        orientation = px_height

    px_conversion = 1 / (orientation / scale_size)

    print(f"1 cm will equate to {1 / px_conversion} pixels width.")

    return (px_conversion)


def classify_surfaces(cont):
    """ Rule based classification of contours based on their size

    Parameters
    ----------
    cont: dataframe
        dataframe with all the contour information and measurements for an image

    Returns
    -------

        A dictionary


    """

    def dorsal_ventral(cont, contours):

        output = [None] * 2
        if (cont[cont['parent_index'] == contours['index'].iloc[0]].shape[0] >
                cont[cont['parent_index'] == contours['index'].iloc[1]].shape[0]):

            output[0] = 'Dorsal'
            output[1] = 'Ventral'
        else:
            output[0] = 'Ventral'
            output[1] = 'Dorsal'

        return output

    surfaces = cont[cont['hierarchy_level'] == 0].copy()  # .sort_values(by=["area_px"], ascending=False)

    names = {}
    # Dorsal, lateral, platform, ventral.
    if surfaces.shape[0] == 1:
        names[0] = 'Dorsal'

    elif surfaces.shape[0] > 1:
        ratio = surfaces["area_px"].iloc[1] / surfaces["area_px"].iloc[0]

        if ratio > 0.9:
            names[0], names[1] = dorsal_ventral(cont, surfaces)

        if surfaces.shape[0] == 2 and ratio <= 0.9:

            if ratio > 0.3:
                names[0] = 'Dorsal'
                names[1] = 'Lateral'
            else:
                names[0] = 'Dorsal'
                names[1] = 'Platform'

        elif surfaces.shape[0] == 3:
            if ratio > 0.9:

                ratio2 = surfaces["area_px"].iloc[2] / surfaces["area_px"].iloc[0]

                if ratio2 > 0.3:
                    names[2] = 'Lateral'
                else:
                    names[2] = 'Platform'
            else:
                names[0] = 'Dorsal'
                names[1] = 'Lateral'
                names[2] = 'Platform'

        elif surfaces.shape[0] == 4:
            names[0], names[1] = dorsal_ventral(cont, surfaces)
            names[2] = 'Lateral'
            names[3] = 'Platform'

        else:
            for i in range(surfaces.shape[0]):
                names[i] = np.nan

    return names


def contour_arrow_classification(cont, hierarchy, quantiles, image_array):
    """

    Function that finds contours that correspond to an arrow.

    Parameters
    ----------
    cont: dataframe
        Dataframe with contour information.
    quantiles
    hierarchy
    Returns
    -------

    a boolean, stating if the contour is likely to be an arrow

    """

    if len(cont) < 50 or len(cont) > 300 or hierarchy < quantiles:
        return False
    else:

        sd = ShapeDetector()

        ratio = 1
        # loop over the contours

        # shape using only the contour
        M = cv2.moments(cont)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)

        shape, vertices = sd.detect(cont)
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image


        if shape == 'arrow':

            return True
        else:
            return False


def find_arrow_templates(image_array, df_contours):
    """

    Decide if a contour is really an arrow and make it into a template

    Parameters
    ----------
    image_array: array
        Array of input image

    df_contours: dataframe
        dataframe with all the contour information and measurements for an image


    Returns
    -------

    """

    templates = []
    index_drop = []

    quantiles = np.quantile([item[-1] for item in df_contours['hierarchy']], 0.2)

    df_contours['arrow'] = False

    for cont, index, hierarchy in df_contours[['contour', 'index','hierarchy']].itertuples(index=False):


        is_arrow = contour_arrow_classification(cont, hierarchy[-1], quantiles, image_array)

        if is_arrow == False:
            continue
        else:

            masked_image = mask_image(image_array, cont, False)

            ratio = len(masked_image[(masked_image > 0.9)]) / len(masked_image[(masked_image != 0)])

            #plot_contour_figure(masked_image, cont)

            if ratio > 0.7:
                continue
            if ratio < 0.3:
                index_drop.append(index)
                continue


            df_contours.loc[df_contours.index == index, 'arrow'] = True


            rows, columns = subtract_masked_image(masked_image)

            new_masked_image = np.delete(image_array, rows[:-3], 0)
            new_masked_image = np.delete(new_masked_image, columns[:-3], 1)

            templates.append(new_masked_image)

    return index_drop, templates


def subtract_masked_image(masked_image_array):
    """

    Given an input masked image, return all rows and columns where
    an image is all masked.


    Parameters
    ----------
    masked_image_array: array
        2D array of an image

    Returns
    -------

    List of columns and rows that have been masked.

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


def template_matching(image_array, templates_df, debug = True):
    """

    Find best template match in an image

    Parameters
    ----------
    image_array: array
        Array of input masked_image_array
    templates_df: array
        Array of template images
    debug: bool
        Make plot if True
    Returns
    -------

    Index of best matching template

    """

    templates = templates_df['template_array'].values

    image_array = image_array.astype(np.float32)

    #TODO: Can we use the intersection of this to only  search inside the contour?
    inner_contour = np.where(image_array != 0)

    location_index = -1

    avg_match = 0
    for i, template in enumerate(templates):
        (tW, tH) = template.shape[::-1]
        (sW, sH) = image_array.shape[::-1]
        if tW > sW or tH>sH:
            continue
        result = cv2.matchTemplate(image_array, template.astype(np.float32), cv2.TM_CCOEFF_NORMED)  # template matching
        threshold = 0.9
        location = np.where(result >= threshold)  # areas where results are >= than threshold value

        #TODO: Invert the loop.
        match = False
        for n1, _ in enumerate(inner_contour[0]):
            (item1,item2) = (inner_contour[0][n1], inner_contour[1][n1])
            for n2, _ in enumerate(location[0]):
                if (location[0][n2], location[1][n2]) == (item1,item2):
                    match = True

        if len(location[0]) > 0 and match==True:
            if result[location].mean() > avg_match:
                index = i
                avg_match = result[location].mean()

    if avg_match>0:
        location_index = index

    if location_index!= -1 and debug:

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        ax[0].imshow(image_array, cmap=plt.cm.gray)
        ax[1].imshow(templates[location_index], cmap=plt.cm.gray)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()
        plt.close(fig)

    return location_index


def get_angles(templates, id):

    template_dict_list = []

    for index, template in enumerate(templates):

        template_dict = {}

        template_dict['id'] = id+"_"+str(index)

        template_dict['template_array'] = template
        template_dict['angle'] = measure_arrow_angle(template)

        template_dict_list.append(template_dict)

    templates_df = pd.DataFrame.from_records(template_dict_list)

    return templates_df



def contour_arrow_selection(df_contours):
    """

    Function that selects contours by their size and removes duplicates.

    Parameters
    ----------
    df_contours: dataframe
        Dataframe with contour information.

    Returns
    -------

    list of indexes

    """



    index_to_drop = []

    for hierarchy_level, index, parent_index, area_px, contour, in df_contours[
        ['hierarchy_level', 'index', 'parent_index', 'area_px', 'contour']].itertuples(index=False):

        pass_selection = True
        if hierarchy_level == 0:
            if area_px / max(df_contours['area_px']) < 0.01:
                pass_selection = False
            if area_px / max(df_contours['area_px']) > 0.4:
                pass_selection = False

        else:
            if area_px / max(df_contours['area_px']) > 0.3:
                pass_selection = False

            # get the total area of the parent figure
            norm = df_contours[df_contours['index'] == parent_index]['area_px'].values[0]
            area = area_px
            percentage = area / norm * 100
            if percentage < 0.2:
                pass_selection = False
            if percentage > 60:
                pass_selection = False


        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax = plt.subplot(111)
        # ax.imshow(image_array, cmap=plt.cm.gray)
        # ax.plot(contour[:, 0], contour[:, 1])

        if pass_selection == False:
            index_to_drop.append(index)

    cent_df = df_contours[['area_px', 'centroid', 'hierarchy_level','contour']]

    import itertools

    for i, j in itertools.combinations(cent_df.index, 2):

        if ((i in index_to_drop) or (j in index_to_drop)):
            continue


        d_ij_centroid = np.linalg.norm(np.asarray(cent_df.loc[i]['centroid']) - np.asarray(cent_df.loc[j]['centroid']))

        if cent_df.loc[i]['area_px'] > cent_df.loc[j]['area_px']:
            ratio = cent_df.loc[j]['area_px'] / cent_df.loc[i]['area_px']
        else:
            ratio = cent_df.loc[i]['area_px'] / cent_df.loc[j]['area_px']


        if d_ij_centroid < 15 and ratio > 0.5:
            if (cent_df.loc[i]['area_px'] < cent_df.loc[j]['area_px']):
                if d_ij_centroid<1:
                    index_to_drop.append(i)
                else:
                    index_to_drop.append(j)
            elif (cent_df.loc[i]['area_px'] > cent_df.loc[j]['area_px']):
                if d_ij_centroid < 1:
                    index_to_drop.append(j)
                else:
                    index_to_drop.append(i)

    return index_to_drop


def measure_arrow_angle(template):

    import math

    #import image and grayscale
    uint_img = np.array(template * 255).astype('uint8')
    gray = 255 - uint_img

    # # Extend the borders for the skeleton
    extended = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT)

    # # Create a copy of the crop for results:
    gray_copy = cv2.cvtColor(extended, cv2.COLOR_GRAY2BGR)
    #
    # Create skeleton of the image
    skeleton = cv2.ximgproc.thinning(extended, None, 1)

    # Threshold the image. White pixels = 0, and black pixels = 10:
    # ret = value used for thresholding, thresh is the image used for thresholding
    retval, thresh = cv2.threshold(skeleton, 128, 10, cv2.THRESH_BINARY)

    # Set the end-points kernel for image convolution

    end_points = np.array([[1, 1, 1],
                           [1, 10, 1],
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
            ordered_points[0] = (point_X, point_Y)
            cv2.circle(gray_copy, (point_X, point_Y), 5, color, -1)
        # Find the tail
        else:
            color = (255, 0, 0)
            ordered_points[1] = (point_X, point_Y)
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
    (dx, dy) = (p1x - p0x, p1y - p0y)
    rads = math.atan2(-dy, dx)
    rads %= 2 * math.pi
    angle_1 = math.degrees(rads)
    print(f"Arrow angle: {angle_1}")



    # Compute the angle
    # angle_2 = math.atan(float(dx)/float(dy))
    # The angle is now in radians (-pi/2 to +pi/2).
    # change to degrees
    # angle_2 *= 180/math.pi
    # print(f"Hypotenuse/opposite angle: {angle_2*-1}")
    # Angle is from from -90 to +90
    # To flip below the line
    # if dy < 0:
    #    angle_2 += 180
    # print(f"Obtuse angle: {angle_2}")

    # # angle derived from endpoints of the line drawn along the arrow
    # line_angle = (180/math.pi)*math.atan(y2/y1)
    # print(print(f"Line angle: {line_angle}"))

    # Show it all
    cv2.imshow("End-Points", gray_copy)
    # cv2.imshow("Extend borders", extended)
    # cv2.imshow("Threshold", thresh)
    cv2.imshow("Skeleton", skeleton)
    # cv2.imshow("Gray", gray)
    cv2.imshow("Binary", binary_image)
    cv2.imshow("Detected Line", detected_line)

    cv2.waitKey(0)

    # returning arrow one for now
    return angle_1










