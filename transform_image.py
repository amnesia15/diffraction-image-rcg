import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required = True,
    help = 'path to load image')
ap.add_argument('-s', '--save', required = True,
    help = 'path to save image')
args = vars(ap.parse_args())

"""1 is upper, 2 is right, 3 is bottom, 4 is left"""
def add_padding(side, number, image):
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

image = cv2.imread(args['path'], cv2.IMREAD_GRAYSCALE)
image_augmented = image.copy()
image_augmented = add_padding(1, 50, image_augmented)
cv2.imshow("Image", image_augmented)
cv2.waitKey(5000)
cv2.imwrite(args['save'], image_augmented)
