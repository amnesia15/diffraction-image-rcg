from keras.models import load_model
import cv2
import argparse
import numpy as np
from finding_center import get_image_for_model

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required = True,
    help = 'path to load image')
args = vars(ap.parse_args())


print('[INFO] loading image...')
image = cv2.imread(args['path'], cv2.IMREAD_GRAYSCALE)
image = get_image_for_model(image)
input_x = image[100, 100:201]
input_x = np.reshape(input_x, (1, 101))

print('[INFO] loading model...')
model = load_model('model_output/model.model')

print('[INFO] predicting...')
preds = model.predict(input_x)

print('[INFO] showing image...')
text = "R = {0:.2f} H = {1:.2f}".format(preds[0, 0], preds[0, 1])
print(text)
output = image.copy()
cv2.putText(output, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
   (255, 255, 0), 1)
cv2.imshow("Image", output)
cv2.waitKey()

