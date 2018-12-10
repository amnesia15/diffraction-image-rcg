import cv2
import os 
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.initializers import glorot_normal
from keras.layers.core import Dropout
from nn_model import NNModel
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--image_path", required = False,
    default = "images/",
    help = "path to input images")
ap.add_argument("-pp", "--par_path", required = False,
    default = "images/params/",
    help = "path to input parameters of images (R, H)")
ap.add_argument("-e", "--epochs", required = False,
    default = 6000,
    help = "number of epochs for training")
ap.add_argument("-lr", "--learning_rate", required = False,
    default = 0.00001,
    help = "learning rate for optimizer")
ap.add_argument("-i", "--iterations", required = False,
    default = 1,
    help = "number of training iteration for the same architecture")
ap.add_argument("-hl", "--hidden_layers", required = False,
    nargs = '+', default = [108, 55],
    help = "number of units per hidden layer")
ap.add_argument("-mo", "--model_output", required = False,
    default = "model_output/",
    help = "path for model outputs")
ap.add_argument("-ts", "--test_split", required = False,
    default = 0.2,
    help = "proportion of the dataset to include in the test")
ap.add_argument("-bs", "--batch_size", required = False,
    default = 32,
    help = "number of samples per gradient update")
ap.add_argument("-dr", "--dropout_rates", required = False,
    nargs = "+", default = [0.3, 0.1],
    help = "droupout rates")

args = vars(ap.parse_args())

dir_imgs = args['image_path']

dir_params = args['par_path']

model_out = args['model_output']


H_LAYERS = [int(x) for x in args["hidden_layers"]]
H_LAYERS = np.array(H_LAYERS)

DROPOUT_RATES = [float(x) for x in args["dropout_rates"]]
DROPOUT_RATES = np.array(DROPOUT_RATES)

TEST_SPL = float(args["test_split"])

BATCH_SIZE = int(args["batch_size"])

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

print("[INFO] spliting data...")

data_x = data[ : ,100, 100:201]
trainX, testX, trainY, testY = train_test_split(data_x, params, test_size = TEST_SPL, random_state = 42)

LEARNING_RATE = float(args["learning_rate"])

#model = NNModel.build(np.array([101, 101, 50, 50, 25]), 101)
#model = NNModel.build(np.array([98, 60]), 101, 0.00001)
#model = NNModel.build(H_LAYERS, 101, LEARNING_RATE)
#model = NNModel.build(np.array([1010,1010, 505, 250, 125, 25]), 101, 0.00001)
model = NNModel.build_dropout(H_LAYERS, 101, LEARNING_RATE, DROPOUT_RATES)

print("[INFO] printing model summary...")

model.summary()

print('[INFO] training model...')

loss_train = []
loss_cv = []

EPOCHS = int(args["epochs"])
ITERATIONS = int(args["iterations"])

mae_hist_train = []
mae_hist_cv = []
mae_hist_test = []

for i in range(0, ITERATIONS):
    #model = NNModel.build(np.array([101, 101, 50, 50, 25]), 101, 0.00001)
    #model = NNModel.build(np.array([1010,1010, 505, 250, 125, 25]), 101, 0.00001)
    #model = NNModel.build(H_LAYERS, 101, LEARNING_RATE)
    model = NNModel.build_dropout(H_LAYERS, 101, LEARNING_RATE, DROPOUT_RATES)
    H = model.fit(trainX, trainY, validation_split=0.25, epochs=EPOCHS, verbose=0,
        batch_size=BATCH_SIZE)
    mae_hist_train.append(H.history['mean_absolute_error'][EPOCHS - 1])
    mae_hist_cv.append(H.history['val_mean_absolute_error'][EPOCHS - 1])
    mae_hist_test.append(model.evaluate(testX, testY)[1])


mae_hist_train_np = np.array(mae_hist_train)
mae_hist_cv_np = np.array(mae_hist_cv)
mae_hist_test_np = np.array(mae_hist_test)

mean_train = np.mean(mae_hist_train_np)
mean_cv = np.mean(mae_hist_cv_np)
mean_test = np.mean(mae_hist_test_np)

std_train = np.std(mae_hist_train_np)
std_cv = np.std(mae_hist_cv_np)
std_test = np.std(mae_hist_test_np)

print("[INFO] printing descriptive statistics...")

file_stat = open('{}stats.txt'.format(model_out), 'w')
file_stat.write("MAE_train = {}  STD_train = {}\n".format(mean_train, std_train))
file_stat.write("MAE_cv = {}  STD_cv = {}\n".format(mean_cv, std_cv))
file_stat.write("MAE_test = {}  STD_test = {}\n".format(mean_test, std_test))
file_stat.close()

# ploting the training for last iteration

print('[INFO] ploting mean absolute error...')
epochs = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(epochs, H.history['mean_absolute_error'], label = "mean_absolute_error_train")
plt.plot(epochs, H.history['val_mean_absolute_error'], label = "mean_absolute_error_cv")
plt.title("Mean absolute error (training, cv, test)")
plt.xlabel("Epoch")
plt.ylabel("Mean absolute error")
plt.legend()
plt.savefig('{}plot.png'.format(model_out))

print('[INFO] evaluating and predicting...')
print("Latest model evaluation is {}.".format(model.evaluate(testX, testY)[1]))
print("Mean model evaluation is {}".format(mean_test))
model.save('{}model.model'.format(model_out))
plot_model(model, to_file='{}model.png'.format(model_out), show_layer_names=True,
    show_shapes=True)

predictions = model.predict(testX)
predictions = np.hstack((predictions, testY))
np.savetxt('{}predictions.txt'.format(model_out), predictions, delimiter='     ', fmt='%.2f')


print('[INFO] ploting scatter plot of predictions...')
plt.figure()
plt.plot(testY[:, 0], predictions[:, 0], 'b.', markersize = 1)
plt.plot([1000, 5000], [1000, 5000], 'r')
plt.xlim((500, 5500))
plt.ylim((500, 5500))
plt.xlabel('Real R')
plt.ylabel('Predicted R')    
plt.title("Predicted vs Real R")
plt.legend()
plt.savefig('{}pred_vs_realR.png'.format(model_out))

plt.figure()
plt.plot(testY[:,1], predictions[:, 1], 'b.', markersize = 1)
plt.plot([100, 10000], [100, 10000], 'r')
plt.xlim((-400, 10500))
plt.ylim((-400, 10500))
plt.xlabel('Real H')
plt.ylabel('Predicted H')
plt.title("Predicted vs Real H")
plt.legend()
plt.savefig('{}pred_vs_realH.png'.format(model_out))

print('[INFO] ploting loss...')
plt.figure()
plt.plot(epochs, H.history['loss'], label = 'loss_training')
plt.plot(epochs, H.history['val_loss'], label = 'loss_cv')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss function (training, cv)")
plt.legend()
plt.savefig('{}loss_func.png'.format(model_out))