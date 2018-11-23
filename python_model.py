import cv2
import os 
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

data_x = data[ : ,101, 100:201]
trainX, testX, trainY, testY = train_test_split(data_x, params, test_size = 0.3, random_state = 42)

model = Sequential()
model.add(Dense(units = 101, activation = 'relu', input_dim = 101))
model.add(Dense(units = 101, activation = 'relu'))
model.add(Dense(units = 50, activation = 'relu'))
model.add(Dense(units = 25, activation = 'relu'))
model.add(Dense(units = 2))

model.compile(loss='mse', optimizer=RMSprop(), metrics=["mean_absolute_error"])


print('[INFO] training model...')

mae_test = []
mae_train = []
mae_cv = []

for i in range(0, 30):
    H = model.fit(trainX, trainY, validation_split=0.2, epochs=1)
    mae_train.append(H.history['mean_absolute_error'])
    mae_cv.append(H.history['val_mean_absolute_error'])
    mae_test.append(model.evaluate(testX, testY)[1])

loss_and_metrics = model.evaluate(testX, testY)


print('[INFO] ploting mean absolute error...')
epochs = np.arange(0, 30)
plt.style.use("ggplot")
plt.figure()
plt.plot(epochs, mae_train, label = "mean_absolute_error_train")
plt.plot(epochs, mae_cv, label = "mean_absolute_error_cv")
plt.plot(epochs, mae_test, label = "mean_absolute_error_test")
plt.title("Mean absolute error (training, cv, test)")
plt.xlabel("Epoch")
plt.ylabel("Mean absolute error")
plt.legend()
plt.savefig("plot.png")

