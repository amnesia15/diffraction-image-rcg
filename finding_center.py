import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required = True,
    help = 'path to load image')
args = vars(ap.parse_args())


def find_center(image):
    # convert the grayscale image to binary image
    thresh = cv2.threshold(image,1,255,0)[1]

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    c_x = int(M["m10"] / M["m00"])
    c_y = int(M["m01"] / M["m00"])

    return (c_x + 1, c_y + 1)

def get_image_for_model(image):
    center = find_center(image)

    row_start = center[1] - 100
    row_end = center[1] + 101
    col_start = center[0] - 100
    col_end = center[0] + 101
    
    new_image = image[row_start:row_end, col_start:col_end]
    
    return new_image

def visualize_centroid(image):
    image = get_image_for_model(image)
    center = find_center(image)
    cv2.circle(image, (center[0], center[1]), 1, (255, 255, 255), -1)
    cv2.putText(image, "centroid", (center[0] - 25, center[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, (255, 255, 255), 1)

    cv2.imshow("Image", image)
    cv2.waitKey(5000)
