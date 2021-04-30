import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import matplotlib
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

    Funtion that selects contours by their size and removes duplicates.

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


        if hierarchy_level == 0:
            if area_px / max(df_contours['area_px']) < 0.05:
                index_to_drop.append(index)
            else:
                continue
        else:
            # get the total area of the parent figure
            norm = df_contours[df_contours['index'] == parent_index]['area_px'].values[0]
            area = area_px
            percentage = area / norm * 100
            if percentage < 0.2:
                index_to_drop.append(index)
            if percentage > 60:
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

    cont_info['lenght'] = len(cont * conversion)
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
        Lenght in mm of the scale

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
    df_contours: dataframe
        Dataframe with contour information.

    Returns
    -------

    a boolean, stating if the contour is likely to be an arrow

    """


    if len(cont) < 50 or len(cont) > 150 or hierarchy < quantiles:
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

            if ratio > 0.6:
                continue
            if ratio < 0.3:
                index_drop.append(index)
                continue

            df_contours.loc[df_contours.index == index, 'arrow'] = True

            rows, columns = subtract_masked_image(masked_image)

            new_masked_image = np.delete(image_array, rows[:-5], 0)
            new_masked_image = np.delete(new_masked_image, columns[:-5], 1)

        # fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        # ax[0].imshow(new_masked_image, cmap=plt.cm.gray)
        # ax[1].imshow(image_array, cmap=plt.cm.gray)
        # ax[1].plot(cont[:, 0], cont[:, 1], label='arrow')
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        # plt.show()
        # plt.close(fig)

            templates.append(new_masked_image)

            matplotlib.image.imsave('template'+str(index)+'.png', new_masked_image)

    return index_drop, templates


def subtract_masked_image(masked_image):
    # Check rows in which all values are equal
    rows = []
    for i in range(masked_image.shape[0]):
        if np.all(masked_image[i] == False):
            rows.append(i)
    # Check Columns in which all values are equal
    columns = []
    trans_masked_image = masked_image.T
    for i in range(trans_masked_image.shape[0]):
        if np.all(trans_masked_image[i] == False):
            columns.append(i)

    return rows, columns


def template_matching(image, templates):

    image = image.astype(np.float32)

    location_index = -1

    avg_match = 0
    for i, template in enumerate(templates):
        (tW, tH) = template.shape[::-1]
        (sW, sH) = image.shape[::-1]
        if tW > sW or tH>sH:
            continue
        result = cv2.matchTemplate(image, template.astype(np.float32), cv2.TM_CCOEFF_NORMED)  # template matching
        threshold = 0.9
        location = np.where(result >= threshold)  # areas where results are >= than threshold value
        if len(location[0]) > 0:
            if result[location].mean() > avg_match:
                index = i
                avg_match = result[location].mean()

    if avg_match>0:
        location_index = index

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    ax[0].imshow(image, cmap=plt.cm.gray)
    if location_index!= -1:
        ax[1].imshow(templates[location_index], cmap=plt.cm.gray)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()
    plt.close(fig)

    return location_index

