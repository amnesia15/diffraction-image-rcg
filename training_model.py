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
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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
ap.add_argument("-r", "--regularization", required = False,
    default = 0,
    help = "regularization (0 - no regularization, 1 - dropout method)")

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

REGULARIZATION = int(args["regularization"])

LEARNING_RATE = float(args["learning_rate"])

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


# Picking up features for a model
data = data[ : ,100, 100:201]


# Scaling the features and output to the range between 0 and 1
sc = MinMaxScaler(feature_range=(0, 1))

data = sc.fit_transform(data)
joblib.dump(sc, '{}data.scaler'.format(model_out))
params = sc.fit_transform(params)
joblib.dump(sc, '{}params.scaler'.format(model_out))


# Building the model with or without regularization
if (REGULARIZATION == 0):
    model = NNModel.build(H_LAYERS, 101, LEARNING_RATE)
elif (REGULARIZATION == 1):
    model = NNModel.build_dropout(H_LAYERS, 101, LEARNING_RATE, DROPOUT_RATES)


print("[INFO] printing model summary...")

model.summary()

loss_train = []
loss_cv = []

EPOCHS = int(args["epochs"])
ITERATIONS = int(args["iterations"])

mae_hist_training = []
mae_hist_test = []

mse_hist_training = []
mse_hist_test = []

mae_error = {
    'r_test': [],
    'h_test': [],
    'r_training': [],
    'h_training': []
}

mse_error = {
    'r_test': [],
    'h_test': [],
    'r_training': [],
    'h_training': []
}

earlystop = EarlyStopping(monitor="val_mean_absolute_error", min_delta=0.0001, patience=5,
    verbose=1, mode="auto")

callback_list = [earlystop]

for i in range(0, ITERATIONS):
    print("[INFO] spliting data...")
    trainX, testX, trainY, testY = train_test_split(data, params, test_size = TEST_SPL)

    if (REGULARIZATION == 0):
        model = NNModel.build(H_LAYERS, 101, LEARNING_RATE)
    elif (REGULARIZATION == 1):
        model = NNModel.build_dropout(H_LAYERS, 101, LEARNING_RATE, DROPOUT_RATES)

    print('[INFO] training model...')

    H = model.fit(trainX, trainY, validation_split=0.25, epochs=EPOCHS, verbose=1,
        batch_size=BATCH_SIZE, callbacks=callback_list)

    
    # Predicting the R and H with the model
    predicted_test = model.predict(testX)
    predicted_training = model.predict(trainX)


    # Loading the normalizer and transforming the values back to normal range
    sc = joblib.load('{}params.scaler'.format(model_out))
    predicted_test_scaled = sc.inverse_transform(predicted_test)
    predicted_training_scaled = sc.inverse_transform(predicted_training)
    testY_scaled = sc.inverse_transform(testY)
    trainY_scaled = sc.inverse_transform(trainY)

    print("SHAPE testY_scaled {} and SHAPE predicted_test_scaled {}".format(testY_scaled.shape, predicted_test_scaled.shape))

    # Calculating the mean absolute error
    current_mae_test = mean_absolute_error(testY_scaled, predicted_test_scaled)
    current_mae_training = mean_absolute_error(trainY_scaled, predicted_training_scaled) 

    # Calculating the mean squared error
    current_mse_test = mean_squared_error(testY_scaled, predicted_test_scaled)
    current_mse_training = mean_squared_error(trainY_scaled, predicted_training_scaled)

    mae_hist_test.append(current_mae_test)
    mae_hist_training.append(current_mae_training)

    mse_hist_test.append(current_mse_test)
    mse_hist_training.append(current_mse_training)

    # Calculating the mean absolute error for R and H for test and training set

    mae_error['r_test'].append(mean_absolute_error(testY_scaled[:, 0], predicted_test_scaled[:, 0]))
    mae_error['h_test'].append(mean_absolute_error(testY_scaled[:, 1], predicted_test_scaled[:, 1]))
    mae_error['r_training'].append(mean_absolute_error(trainY_scaled[:, 0], predicted_training_scaled[:, 0]))
    mae_error['h_training'].append(mean_absolute_error(trainY_scaled[:, 1], predicted_training_scaled[:, 1]))

    # Calculating the mean squared error for R and H for test and training set

    mse_error['r_test'].append(mean_squared_error(testY_scaled[:, 0], predicted_test_scaled[:, 0]))
    mse_error['h_test'].append(mean_squared_error(testY_scaled[:, 1], predicted_test_scaled[:, 1]))
    mse_error['r_training'].append(mean_squared_error(trainY_scaled[:, 0], predicted_training_scaled[:, 0]))
    mse_error['h_training'].append(mean_squared_error(trainY_scaled[:, 1], predicted_training_scaled[:, 1]))



# Converting the list to numpy array
mae_hist_test = np.array(mae_hist_test)
mae_hist_training = np.array(mae_hist_training)

mse_hist_test = np.array(mse_hist_test)
mse_hist_training = np.array(mse_hist_training)

mae_error['r_test'] = np.array(mae_error['r_test'])
mae_error['h_test'] = np.array(mae_error['h_test'])
mae_error['r_training'] = np.array(mae_error['r_training'])
mae_error['h_training'] = np.array(mae_error['h_training']) 

mse_error['r_test'] = np.array(mse_error['r_test'])
mse_error['h_test'] = np.array(mse_error['h_test'])
mse_error['r_training'] = np.array(mse_error['r_training'])
mse_error['h_training'] = np.array(mse_error['h_training'])


# Calculating the mean value for the errors
mean_mae_test = np.mean(mae_hist_test)
mean_mae_training = np.mean(mae_hist_training)

mean_mse_test = np.mean(mse_hist_test)
mean_mse_training = np.mean(mse_hist_training)

mae_error['r_test_mean'] = np.mean(mae_error['r_test'])
mae_error['h_test_mean'] = np.mean(mae_error['h_test'])
mae_error['r_training_mean'] = np.mean(mae_error['r_training'])
mae_error['h_training_mean'] = np.mean(mae_error['h_training'])

mse_error['r_test_mean'] = np.mean(mse_error['r_test'])
mse_error['h_test_mean'] = np.mean(mse_error['h_test'])
mse_error['r_training_mean'] = np.mean(mse_error['r_training'])
mse_error['h_training_mean'] = np.mean(mse_error['h_training'])

# Calculating the standard deviation for the errors
std_mae_test = np.std(mae_hist_test)
std_mae_training = np.std(mae_hist_training)

std_mse_test = np.std(mse_hist_test)
std_mse_training = np.std(mse_hist_training)

mae_error['r_test_std'] = np.std(mae_error['r_test'])
mae_error['h_test_std'] = np.std(mae_error['h_test'])
mae_error['r_training_std'] = np.std(mae_error['r_training'])
mae_error['h_training_std'] = np.std(mae_error['h_training'])

mse_error['r_test_std'] = np.std(mse_error['r_test'])
mse_error['h_test_std'] = np.std(mse_error['h_test'])
mse_error['r_training_std'] = np.std(mse_error['r_training'])
mse_error['h_training_std'] = np.std(mse_error['h_training'])


print("[INFO] printing descriptive statistics...")

file_stat = open('{}stats.txt'.format(model_out), 'w')
file_stat.write("MAE_train = {}\t STD_train = {}\n".format(mean_mae_training, std_mae_training))
file_stat.write("MAE_test = {}\t STD_test = {}\n".format(mean_mae_test, std_mae_test))
file_stat.write('-----------------------------------\n')
file_stat.write("MAE_train (R) = {}\t STD_train (R) = {}\n".format(mae_error['r_training_mean'], mae_error['r_training_std']))
file_stat.write("MAE_train (H) = {}\t STD_train (H) = {}\n".format(mae_error['h_training_mean'], mae_error['h_training_std']))
file_stat.write("MAE_test (R) = {}\t STD_test (R) = {}\n".format(mae_error['r_test_mean'], mae_error['r_test_std']))
file_stat.write("MAE_test (H) = {}\t STD_test (H) = {}\n".format(mae_error['h_test_mean'], mae_error['h_test_std']))

file_stat.write('\n************************************\n\n')

file_stat.write("MSE_train = {}  STD_train = {}\n".format(mean_mse_training, std_mse_training))
file_stat.write("MSE_test = {}  STD_test = {}\n".format(mean_mse_test, std_mse_test))
file_stat.write('-----------------------------------\n')
file_stat.write("MSE_train (R) = {}\t STD_train (R) = {}\n".format(mse_error['r_training_mean'], mse_error['r_training_std']))
file_stat.write("MSE_train (H) = {}\t STD_train (H) = {}\n".format(mse_error['h_training_mean'], mse_error['h_training_std']))
file_stat.write("MSE_test (R) = {}\t STD_test (R) = {}\n".format(mse_error['r_test_mean'], mse_error['r_test_std']))
file_stat.write("MSE_test (H) = {}\t STD_test (H) = {}\n".format(mse_error['h_test_mean'], mse_error['h_test_std']))

file_stat.close()

# Ploting the training plot for the last iteration
print('[INFO] ploting mean absolute error...')
epochs = np.arange(0, len(H.history['mean_absolute_error']))
plt.style.use("ggplot")
plt.figure()
plt.plot(epochs, H.history['mean_absolute_error'], label = "mean_absolute_error_train")
plt.plot(epochs, H.history['val_mean_absolute_error'], label = "mean_absolute_error_cv")
plt.title("Mean absolute error (training, cv)")
plt.xlabel("Epoch")
plt.ylabel("Mean absolute error")
plt.legend()
plt.savefig('{}plot.png'.format(model_out))

print('[INFO] printing model architecture and predictions...')

# Saving the model and printing the architecture of the model
model.save('{}model.model'.format(model_out))
plot_model(model, to_file='{}model.png'.format(model_out), show_layer_names=True,
    show_shapes=True)

# Saving the predictions of the model from the last iteration
predictions = np.hstack((predicted_test_scaled, testY_scaled))
np.savetxt('{}predictions.txt'.format(model_out), predictions, delimiter='     ', fmt='%.2f')


print('[INFO] ploting scatter plot of predictions...')

# Ploting the scatter plot of predictions for R
plt.figure()
plt.plot(testY_scaled[:, 0], predictions[:, 0], 'b.', markersize = 1)
plt.plot([1000, 5000], [1000, 5000], 'r')
plt.xlim((500, 5500))
plt.ylim((500, 5500))
plt.xlabel('Real R')
plt.ylabel('Predicted R')    
plt.title("Predicted vs Real R")
plt.legend()
plt.savefig('{}pred_vs_realR.png'.format(model_out))

#Ploting the scatter plot of prediction for H
plt.figure()
plt.plot(testY_scaled[:,1], predictions[:, 1], 'b.', markersize = 1)
plt.plot([100, 10000], [100, 10000], 'r')
plt.xlim((-400, 10500))
plt.ylim((-400, 10500))
plt.xlabel('Real H')
plt.ylabel('Predicted H')
plt.title("Predicted vs Real H")
plt.legend()
plt.savefig('{}pred_vs_realH.png'.format(model_out))

print('[INFO] ploting loss...')

#Ploting the loss function from the last iteration
plt.figure()
plt.plot(epochs, H.history['loss'], label = 'loss_training')
plt.plot(epochs, H.history['val_loss'], label = 'loss_cv')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss function (training, cv)")
plt.legend()
plt.savefig('{}loss_func.png'.format(model_out))