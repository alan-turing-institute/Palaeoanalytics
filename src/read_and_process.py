import skimage
import skimage.feature
import skimage.viewer

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


