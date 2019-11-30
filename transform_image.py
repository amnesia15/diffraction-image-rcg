import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
                help='path to load image')
ap.add_argument('-s', '--save', required=True,
                help='path to save image')
args = vars(ap.parse_args())


def add_padding(side, number, image):
    """Adds empty row or column of pixels to the image.

    # Arguments
        side: sides to which pixels will be added. `1` stands
            for upper side, `2` for right side, `3` is for bottom
            and `4` is for left side.
        number: number of rows or columns of pixels to be added.
        image: ndarray of pixel values.

    # Returns
        Transformed image with padding of pixels. An instance of
        numpy ndarray.

    # Raises
        ValueError: in case an invalid value for `side` or `number`
            argument is passed.
    """
    if side < 1 or side > 4:
        raise ValueError('side argument is not in range between 1 '
                         'and 4')

    if number < 0:
        raise ValueError('number argument cannot be less than 0')

    if (side == 1):
        zeros = np.zeros((number, image.shape[1]), dtype=np.uint8)
        image = np.vstack((zeros, image))
    elif (side == 2):
        zeros = np.zeros((image.shape[0], number), dtype=np.uint8)
        image = np.hstack((image, zeros))
    elif (side == 3):
        zeros = np.zeros((number, image.shape[1]), dtype=np.uint8)
        image = np.vstack((image, zeros))
    elif side == 4:
        zeros = np.zeros((image.shape[0], number), dtype=np.uint8)
        image = np.hstack((zeros, image))
    else:
        print('Error!')
        exit(1)
    return image


if __name__ == '__main__':
    image = cv2.imread(args['path'], cv2.IMREAD_GRAYSCALE)
    image_augmented = image.copy()
    image_augmented = add_padding(1, 50, image_augmented)
    cv2.imshow("Image", image_augmented)
    cv2.waitKey(5000)
    cv2.imwrite(args['save'], image_augmented)
