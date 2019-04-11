from keras.models import load_model
import cv2
import argparse
import numpy as np
from finding_center import get_image_for_model
from sklearn.externals import joblib

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required = True,
    help = 'path to load image')
ap.add_argument('-ds', '--data_scaler', required = False,
    default = "model_output/data.scaler",
    help = 'path to scalar for features')
ap.add_argument('-ps', '--params_scaler', required = False,
    default = "model_output/params.scaler",
    help = 'path to load scalar for output variables')
ap.add_argument("-mo", "--model_output", required = False,
    default = "model_output/",
    help = "path for model outputs")
args = vars(ap.parse_args())

path_params_scalar = args['params_scaler']
path_data_scalar = args['data_scaler']

print('[INFO] loading image...')
image = cv2.imread(args['path'], cv2.IMREAD_GRAYSCALE)
image = get_image_for_model(image)
input_x = image[100, 100:201]
input_x = np.reshape(input_x, (1, 101))

sc_data = joblib.load(path_data_scalar)
input_x = sc_data.transform(input_x)

print('[INFO] loading model...')
model = load_model('model_output/model.model')

print('[INFO] predicting...')
preds = model.predict(input_x)

sc_params = joblib.load(path_params_scalar)
preds = sc_params.inverse_transform(preds)
print('[INFO] showing image...')
text = "R = {0:.2f} H = {1:.2f}".format(preds[0, 0], preds[0, 1])
print(text)
output = image.copy()
cv2.putText(output, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
   (255, 255, 0), 1)
cv2.imwrite("{}prediction_output.png".format(args['model_output']), output)
