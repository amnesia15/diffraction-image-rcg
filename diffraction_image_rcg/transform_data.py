"""Script for transforming the output from the Fortan modeling program
to the txt file that contains pixels.

Args:
    -dp: path to the directory that contains raw data.
    -op: path to the directory where we want to output the transformed data.
Returns:
    None. Saves the transformed data.
"""
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "-dp",
    "--data_path",
    required=False,
    default="../resources/images/raw_data/",
    help="path to input data",
)
ap.add_argument(
    "-op",
    "--output_path",
    required=False,
    default="../resources/output/",
    help="path to output images",
)
args = vars(ap.parse_args())


data = []
no_images = 1000

for i in range(0, no_images):
    data_path = args["data_path"] + "input" + str(i + 1) + "/DIFSLIKA.TXT"

    cur_data = np.loadtxt(data_path)

    data.append(cur_data)

data = np.asarray(data)

max_lum = np.max(data[:, :, 2])

data[:, :, 2] /= max_lum

data_indexes = np.asarray(data[:, :, 0:2], dtype=int)

data_indexes[:, :, :] = (data_indexes[:, :, :] + 10000) / 100

img_mat = np.zeros((1000, 201, 201))

for i in range(0, no_images):
    for j in range(0, data_indexes.shape[1]):
        img_mat[i, data_indexes[i, j, 0], data_indexes[i, j, 1]] = data[i, j, 2]

for i in range(0, no_images):
    out_path = args["output_path"] + "SLIKA" + str(i + 1) + ".txt"
    np.savetxt(out_path, img_mat[i, :, :])
