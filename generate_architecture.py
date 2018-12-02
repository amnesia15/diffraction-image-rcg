from sklearn.model_selection import train_test_split
from nn_model import NNModel
import cv2
import numpy as np

dir_imgs = '/home/amnesia/Desktop/1000_imgs/'

dir_params = '/home/amnesia/Desktop/1000/'


data = []
params = []

print("[INFO] loading images...")

for i in range(0, 1000):
    img_path = dir_imgs + "SLIKA" + str(i + 1) + ".png"
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    data.append(image)
    param_path = dir_params + 'input' + str(i + 1) + '/INPUT.TXT'
    params.append(np.loadtxt(param_path))

data = np.array(data)
params = np.array(params)

print("[INFO] creating models")

data_x = data[ : ,100, 100:201]
trainX, testX, trainY, testY = train_test_split(data_x, params, test_size = 0.2, random_state = 42)

low_high_arr = np.array([90, 110, 40, 60, 15, 35]).reshape((3, 2))
units = NNModel.generate_combination_low_high_different(low_high_arr)
print("Number of combinations = {}".format(units.shape[0]))