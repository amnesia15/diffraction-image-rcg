import cv2
import argparse


def find_center(image):
    """Function responsible for finding center of mass for the
    diffraction image.

    # Args
        image: ndarray of pixel values.

    # Returns
        Tuple of indexes that represents center of mass of the image.
        Each tuple value represents the x and y axis index.
    """
    # convert the grayscale image to binary image
    thresh = cv2.threshold(image, 1, 255, 0)[1]

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    c_x = int(M["m10"] / M["m00"])
    c_y = int(M["m01"] / M["m00"])

    return (c_x + 1, c_y + 1)


def prepare_model_image(image):
    """Function returns image suitable for model training.
    Finds center of mass for the diffraction image and then
    cuts the image of the size 201 x 201 from the center.

    # Args
        image: ndarray of pixel values.

    # Returns
        Image suitable for the training with 201 x 201 dimensions.
    """
    center = find_center(image)

    row_start = center[1] - 100
    row_end = center[1] + 101
    col_start = center[0] - 100
    col_end = center[0] + 101

    cropped_image = image[row_start:row_end, col_start:col_end]

    return cropped_image


def visualize_centroid(image):
    """Functions shows the diffraction image with the indicator
    of location of the centroid.

    # Args
        image: ndarray of pixel values.

    # Returns
        None.
    """
    image = prepare_model_image(image)
    center = find_center(image)
    cv2.circle(image, (center[0], center[1]), 1, (255, 255, 255), -1)
    cv2.putText(
        image,
        "centroid",
        (center[0] - 25, center[1] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Image", image)
    cv2.waitKey(5000)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path to load image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["path"], cv2.IMREAD_GRAYSCALE)

    visualize_centroid(image)
