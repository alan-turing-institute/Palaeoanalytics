import matplotlib.pyplot as plt
import pylithics.src.utils as utils
import os
import warnings

def fig_size(image_array):
    """

    Calculate optimum figure horizontal width based on the ratio of sizes of rows and columns.

    Parameters
    ----------
    image_array: array
        Original image array (0 to 255)
    Returns
    -------
    A number for the x width of the figure size.
    """

    ratio = image_array.shape[0] / image_array.shape[1]
    if ratio < 0.7:
        x_size = 20 / ratio
    elif ratio < 1:
        x_size = 22 / ratio
    elif ratio < 1.2:
        x_size = 22 * ratio
    else:
        x_size = 18 * ratio

    return x_size

def plot_surfaces(image_array, contours_df, output_figure):
    """
    Plot the contours from the lithic surfaces.

    Parameters
    ----------
    image_array: array
        array of an unprocessed image read by openCV (0, 255 pixels)
    contours_df: dataframe
        Dataframe with detected contours and extra information about them.
    output_figure: str
        path including name of the figure to be saved

    """
    fig_x_size = fig_size(image_array)
    fig, ax = plt.subplots(figsize=(fig_x_size, 20))
    ax.imshow(image_array, cmap=plt.cm.gray)

    surfaces_classification = utils.classify_surfaces(contours_df)
    # selecting only surfaces (lowest hierarchy level).
    contours_surface_df = contours_df[contours_df['parent_index'] == -1].sort_values(by=["area_px"], ascending=False)

    if contours_surface_df.shape[0] == 0:
        warnings.warn("Warning: No surfaces detected, no surface output figure will be saved.'")
        return None

    cmap_list = plt.cm.get_cmap('Paired', contours_surface_df.shape[0])

    i = 0
    for contour in contours_surface_df['contour'].values:
        classification = surfaces_classification[i]
        text = str(classification)
        ax.plot(contour[:, 0], contour[:, 1], label=text, linewidth=10, color=cmap_list(i))
        i = i + 1

    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=17)
    plt.title("Detected surfaces", fontsize=30)
    plt.savefig(output_figure)
    plt.close(fig)


def plot_scars(image_array, contours_df, output_figure):
    """
    Plot the contours from the lithic surfaces.

    Parameters
    ----------
    image_array: array
        Original image array (0 to 255)
    contours_df: dataframe
        Dataframe with detected contours and extra information about them.
    output_figure: str
        Path including name of the figure to be saved

    """
    fig_x_size = fig_size(image_array)
    fig, ax = plt.subplots(figsize=(fig_x_size, 20))

    ax.imshow(image_array, cmap=plt.cm.gray)

    # selecting only surfaces (lowest hiearchy level).
    contours_scars_df = contours_df[contours_df['parent_index'] != -1].sort_values(by=["area_px"], ascending=False)

    if contours_scars_df.shape[0] == 0:
        warnings.warn("Warning: No scars detected, no scar output figure will be saved.'")
        return None

    cmap_list = plt.cm.get_cmap('tab20', contours_scars_df.shape[0])

    i = 0
    for contour, area_mm, width_mm, height_mm in \
            contours_scars_df[['contour', 'area_mm',
                                 'width_mm', 'height_mm']].itertuples(index=False):
        text = "A: " + str(area_mm) + ", B: " + str(width_mm) + ", L: " + str(height_mm)
        ax.plot(contour[:, 0], contour[:, 1], label=text, linewidth=5, color=cmap_list(i))
        i = i + 1

    ax.set_xticks([])
    ax.set_yticks([])
    plt.figtext(0.02, 0.5, ("A: Total Area"), fontsize=18)
    plt.figtext(0.02, 0.52, ("B: Maximum Breadth"), fontsize=18)
    plt.figtext(0.02, 0.54, ("L: Maximum Length"), fontsize=18)
    plt.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=11)
    plt.title("Scar measurements (in millimeters)", fontsize=30)
    plt.savefig(output_figure)
    plt.close(fig)

def plot_angles(image_array, contours_df, output_path):
    """
    Plot the contours from the lithic surfaces.

    Parameters
    ----------
    image_array: array
        Original image array (0 to 255)
    contours_df: dataframe
        Dataframe with detected contours and extra information about them.
    output_path: str
        Path to output directory to save processed images

    """
    fig_x_size = fig_size(image_array)
    fig, ax = plt.subplots(figsize=(fig_x_size, 20))
    ax.imshow(image_array, cmap=plt.cm.gray)

    # selecting only scars with angles
    contours_angles_df = contours_df[(contours_df['parent_index'] != -1) & (contours_df['angle'].notnull())].sort_values(by=["area_px"], ascending=False)
    cmap_list = plt.cm.get_cmap('tab20', contours_angles_df.shape[0])

    if contours_angles_df.shape[0] == 0:
        warnings.warn("Warning: No scars with measured angles detected, no angle output figure will be saved.'")
        return None

    i = 0
    for contour, angle in \
            contours_angles_df[['contour', 'angle']].itertuples(index=False):
        text = "Angle: " + str(angle)
        ax.plot(contour[:, 0], contour[:, 1], label=text, linewidth=5, color=cmap_list(i))
        i = i + 1

    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=11)
    plt.title("Scar Strike Angle measurement (in degrees)", fontsize=30)
    plt.savefig(output_path)
    plt.close(fig)


def plot_results(id, image_array, contours_df, output_dir):
    """
    Plot the results of the object characterisation.

    Parameters
    ----------
    id: str
        Name of the lithic
    image_array: array
         Original image array (0 to 255)
    contours_df: dataframe
         Dataframe with detected contours and extra information about them.
    output_figure: str
        Path to output directory to save processed images

    Returns
    -------
    an array
    """

    # plot surfaces
    output_lithic = os.path.join(output_dir, id + "_lithic_surfaces.png")
    plot_surfaces(image_array, contours_df, output_lithic)

    # plot scars
    output_lithic = os.path.join(output_dir, id + "_lithium_scars.png")
    plot_scars(image_array, contours_df, output_lithic)

    # plot scar strike angle
    output_lithic = os.path.join(output_dir, id + "_lithium_angles.png")
    plot_angles(image_array, contours_df, output_lithic)

    # plot scar strike angle
    output_lithic = os.path.join(output_dir, id + "_complexity_polygon_count.png")
    plot_complexity(image_array, contours_df, output_lithic)



def plot_thresholding(image_array, threshold, binary_array, output_file=''):
    """
    Visualize the effect of the thresholding on images. Produces three plots, l:r1) original,
    2) pixel intensity histogram, 3) thresholded image.

    Parameters
    ----------
    image_array:  array
        Array of an unprocessed image read by openCV (0:255 pixels)
    threshold: float
        Threshold value found for images
    binary_array: array
        Resulting binary image with pixel values of 0,1
    Returns
    -------
    an array
    """

    text = 'edge detection'
    fig, axes = plt.subplots(ncols=3, figsize=(10, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(image_array, cmap=plt.cm.gray)
    ax[0].set_title('Processed original')
    ax[0].axis('off')

    ax[1].hist(image_array.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(threshold, color='r')

    ax[2].imshow(binary_array, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded and ' + text)
    ax[2].axis('off')

    if output_file != "":
        plt.savefig(output_file)
    plt.close(fig)


def plot_contour_figure(image_array, cont):
    """
    Returns plots of image and a give contour.

    Parameters
    ----------
    image_array: array
        Array of an unprocessed image read by openCV (0:255 pixels)
    cont: list
        Array of coordinates for a contour
    Returns
    -------
    an array
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.imshow(image_array, cmap=plt.cm.gray)
    ax.plot(cont[:, 0], cont[:, 1])
    plt.close(fig)


def plot_template_arrow(image_array, template_array, value):
    """
    Plot arrows for associated scars.

    Parameters
    ----------
    image_array: array
        Array of an unprocessed image read by openCV (0:255 pixels)
    template_array: array
        Array of an template image  (0:255 pixels)
    Returns
    -------

    """

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    ax[0].imshow(image_array, cmap=plt.cm.gray)
    ax[1].imshow(template_array, cmap=plt.cm.gray)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.figtext(0.4, 0.9, str(value))
    plt.show()

def plot_complexity(image_array, contours_df, output_path):
    """
    Plot the contours from the lithic surfaces and display complexity and polygon count measurements.

    Parameters
    ----------
    image_array: array
        Original image array (0 to 255)
    contours_df: dataframe
        Dataframe with detected contours and extra information about them.
    output_path: str
        Path to output directory to save processed images

    """
    fig_x_size = fig_size(image_array)
    fig, ax = plt.subplots(figsize=(fig_x_size, 20))
    ax.imshow(image_array, cmap=plt.cm.gray)

    # selecting only scars with a complexity measure > 0
    contours_complexity_df = contours_df[(contours_df['parent_index'] != -1) & (contours_df['complexity']>0)]
    cmap_list = plt.cm.get_cmap('tab20', contours_complexity_df.shape[0])

    if contours_complexity_df.shape[0] == 0:
        warnings.warn("Warning: No scars with complexity measure, no complexity output figure will be saved.'")
        return None

    i = 0
    for contour, complexity, polygon_count in \
            contours_complexity_df[['contour', 'complexity','polygon_count']].itertuples(index=False):
        text = "Complexity: " + str(complexity)+", Polygon Count: "+str(polygon_count)
        ax.plot(contour[:, 0], contour[:, 1], label=text, linewidth=5, color=cmap_list(i))
        i = i + 1

    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0, fontsize=11)
    plt.title("Scar complexity and polygon count measurements", fontsize=30)
    plt.savefig(output_path)
    plt.close(fig)
