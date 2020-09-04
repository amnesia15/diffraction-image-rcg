from sklearn.model_selection import train_test_split
from nn_model import NNModel
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "-ip",
    "--image_path",
    required=False,
    default="images/",
    help="path to input images",
)
ap.add_argument(
    "-pp",
    "--par_path",
    required=False,
    default="images/params/",
    help="path to input parameters of images (R, H)",
)
ap.add_argument(
    "-e", "--epochs", required=False, default=50, help="number of epochs for training"
)
ap.add_argument(
    "-lr",
    "--learning_rate",
    required=False,
    default=0.00001,
    help="learning rate for optimizer",
)
ap.add_argument(
    "-mo",
    "--model_output",
    required=False,
    default="model_output/",
    help="path for model outputs",
)
ap.add_argument(
    "-ul",
    "--units_limit",
    required=False,
    nargs="+",
    default=[90, 110, 40, 60],
    help="limits per hidden layer",
)
ap.add_argument(
    "-ts",
    "--test_split",
    required=False,
    default=0.2,
    help="proportion of the dataset to include in the test",
)
ap.add_argument(
    "-bs",
    "--batch_size",
    required=False,
    default=32,
    help="number of samples per gradient update",
)
ap.add_argument(
    "-dr",
    "--dropout_rates",
    required=False,
    nargs="+",
    default=[0.3, 0.1],
    help="droupout rates",
)
ap.add_argument(
    "-r",
    "--regularization",
    required=False,
    default=0,
    help="regularization (0 - no regularization, " "1 - dropout method)",
)

args = vars(ap.parse_args())

dir_imgs = args["image_path"]

dir_params = args["par_path"]

LIM_UNITS = [int(x) for x in args["units_limit"]]
LIM_UNITS = np.array(LIM_UNITS)
if LIM_UNITS.size % 2 != 0:
    print("-ul not multiple of 2")
    exit(1)
LIM_UNITS = LIM_UNITS.reshape(LIM_UNITS.size / 2, 2)

DROPOUT_RATES = [float(x) for x in args["dropout_rates"]]
DROPOUT_RATES = np.array(DROPOUT_RATES)

LEARNING_RATE = float(args["learning_rate"])

TEST_SPL = float(args["test_split"])

BATCH_SIZE = int(args["batch_size"])

REGULARIZATION = int(args["regularization"])

data = []
params = []

print("[INFO] loading images...")

for i in range(0, 1000):
    img_path = dir_imgs + "SLIKA" + str(i + 1) + ".png"
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    data.append(image)
    param_path = dir_params + "input" + str(i + 1) + "/INPUT.TXT"
    params.append(np.loadtxt(param_path))

data = np.array(data)
params = np.array(params)

print("[INFO] creating models")

data_x = data[:, 100, 100:201]
trainX, testX, trainY, testY = train_test_split(
    data_x, params, test_size=TEST_SPL, random_state=42
)

# low_high_arr = np.array([90, 110, 40, 60, 15, 35]).reshape((3, 2))
# low_high_arr = np.array([90, 110, 40, 60]).reshape((2, 2))

units = NNModel.generate_combination_low_high_different(LIM_UNITS)
print("Number of combinations = {}".format(units.shape[0]))

EPOCHS = int(args["epochs"])
mae_hist_train = []
mae_hist_cv = []
mae_hist_test = []

if REGULARIZATION == 0:
    model = NNModel.build(
        units[
            0,
        ],
        101,
        LEARNING_RATE,
    )
elif REGULARIZATION == 1:
    model = NNModel.build_dropout(
        units[
            0,
        ],
        101,
        LEARNING_RATE,
        DROPOUT_RATES,
    )

H = model.fit(
    trainX,
    trainY,
    validation_split=0.25,
    epochs=EPOCHS,
    verbose=0,
    batch_size=BATCH_SIZE,
)
mae_hist_train.append(H.history["mean_absolute_error"][EPOCHS - 1])
mae_hist_cv.append(H.history["val_mean_absolute_error"][EPOCHS - 1])
mae_hist_test.append(model.evaluate(testX, testY)[1])

best_mae = mae_hist_test[0]
best_hidden = units[
    0,
]


for i in range(1, units.shape[0]):
    print("Iteration no. {}".format(i))
    if REGULARIZATION == 0:
        model = NNModel.build(
            units[
                0,
            ],
            101,
            LEARNING_RATE,
        )
    elif REGULARIZATION == 1:
        model = NNModel.build_dropout(
            units[
                0,
            ],
            101,
            LEARNING_RATE,
            DROPOUT_RATES,
        )

    H = model.fit(
        trainX,
        trainY,
        validation_split=0.25,
        epochs=EPOCHS,
        verbose=0,
        batch_size=BATCH_SIZE,
    )

    mae_hist_train.append(H.history["mean_absolute_error"][EPOCHS - 1])
    mae_hist_cv.append(H.history["val_mean_absolute_error"][EPOCHS - 1])
    mae_hist_test.append(model.evaluate(testX, testY)[1])

    if mae_hist_test[i] < best_mae:
        best_mae = mae_hist_test[i]
        best_hidden = units[
            i,
        ]


file = open("{}best.txt".format(args["model_output"]), "w")
file.write("Best mae: {}\n".format(best_mae))
file.write("Hidden units: ")
for i in range(0, best_hidden.size):
    file.write("{} ".format(best_hidden[i]))
file.close()
