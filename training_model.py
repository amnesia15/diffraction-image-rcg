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

print("[INFO] creating model...")

data_x = data[ : ,100, 100:201]
trainX, testX, trainY, testY = train_test_split(data_x, params, test_size = 0.2, random_state = 42)


model = NNModel.build(np.array([101, 101, 50, 50, 25]), 101)

model.summary()

print('[INFO] training model...')

mae_test = []
mae_train = []
mae_cv = []

loss_train = []
loss_cv = []

EPOCHS = 5000

for i in range(0, EPOCHS):
    H = model.fit(trainX, trainY, validation_split=0.25, epochs=1)
    mae_train.append(H.history['mean_absolute_error'])
    mae_cv.append(H.history['val_mean_absolute_error'])
    mae_test.append(model.evaluate(testX, testY)[1])
    loss_train.append(H.history['loss'])
    loss_cv.append(H.history['val_loss'])

loss_and_metrics = model.evaluate(testX, testY)


print('[INFO] ploting mean absolute error...')
epochs = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(epochs, mae_train, label = "mean_absolute_error_train")
plt.plot(epochs, mae_cv, label = "mean_absolute_error_cv")
plt.plot(epochs, mae_test, label = "mean_absolute_error_test")
plt.title("Mean absolute error (training, cv, test)")
plt.xlabel("Epoch")
plt.ylabel("Mean absolute error")
plt.legend()
plt.savefig('model_output/plot.png')

print('[INFO] evaluating and predicting...')
print(model.evaluate(testX, testY))
model.save('model_output/model.model')
plot_model(model, to_file='model_output/model.png', show_layer_names=True,
    show_shapes=True)

predictions = model.predict(testX)
predictions = np.hstack((predictions, testY))
np.savetxt('model_output/predictions.txt', predictions, delimiter='     ', fmt='%.2f')


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
plt.savefig('model_output/pred_vs_realR.png')

plt.figure()
plt.plot(testY[:,1], predictions[:, 1], 'b.', markersize = 1)
plt.plot([100, 10000], [100, 10000], 'r')
plt.xlim((-400, 10500))
plt.ylim((-400, 10500))
plt.xlabel('Real H')
plt.ylabel('Predicted H')
plt.title("Predicted vs Real H")
plt.legend()
plt.savefig('model_output/pred_vs_realH.png')

print('[INFO] ploting loss...')
plt.figure()
plt.plot(epochs, loss_train, label = 'loss_training')
plt.plot(epochs, loss_cv, label = 'loss_cv')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss function (training, cv)")
plt.legend()
plt.savefig('model_output/loss_func.png')