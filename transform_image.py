import cv2
import argparse
import numpy as np


def add_padding(image, side, number):
    """Adds empty rows or columns of pixels to the image.

    # Arguments
        image: ndarray of pixel values.
        side: sides to which pixels will be added. `top` stands
            for upper side, `right` for right side, `bottom` is for bottom
            and `left` is for left side.
        number: number of rows or columns of pixels to be added.

    # Returns
        Transformed image with padding of pixels. An instance of
        numpy ndarray.

    # Raises
        ValueError: in case an invalid value for `side` or `number`
            argument is passed.
    """
    if not isinstance(number, int):
        raise TypeError('number argument is not intenger')

    possible_sides = ['top', 'right', 'bottom', 'left']
    if side not in possible_sides:
        raise ValueError('side argument is invalid. side argument '
                         'can take only one of the following values: '
                         'top, right, bottom, left')

    if number < 0:
        raise ValueError('number argument cannot be less than 0')

    if (side == 'top'):
        zeros = np.zeros((number, image.shape[1]), dtype=np.uint8)
        image = np.vstack((zeros, image))
    elif (side == 'right'):
        zeros = np.zeros((image.shape[0], number), dtype=np.uint8)
        image = np.hstack((image, zeros))
    elif (side == 'bottom'):
        zeros = np.zeros((number, image.shape[1]), dtype=np.uint8)
        image = np.vstack((image, zeros))
    elif side == 'left':
        zeros = np.zeros((image.shape[0], number), dtype=np.uint8)
        image = np.hstack((zeros, image))

    return image


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path',
                    required=True,
                    help='path to load image')
    ap.add_argument('-s', '--save',
                    required=True,
                    help='path to save image')
    ap.add_argument('-side', '--side',
                    default='top',
                    help='sides to which pixels will be added (top - '
                         'upper side, right - right side, bottom - bottom '
                         'side and left - left side')
    ap.add_argument('-number', '--number',
                    default=50,
                    help='number of rows or columns of pixels to be added')

    args = vars(ap.parse_args())

    # Load the image from the certain location,
    # transform it and save it on the provided location
    image = cv2.imread(args['path'], cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    image_augmented = image.copy()
    image_augmented = add_padding(image_augmented,
                                  args['side'],
                                  int(args['number']))
    cv2.imshow('Image', image_augmented)
    cv2.waitKey(5000)
    cv2.imwrite(args['save'], image_augmented)
