""" Script for training the model.

# Args
    -ip: Path to the directory that contains the input images.
    -pp: Path to the directory that contains params of the images.
    -e: Number of epochs for training.
    -lr: Learning rate for the optimizer.
    -i: Number of training iteration for the same architecture.
    -hl: Number of units per hidden layer.
    -mo: The directory for the outputs of the model.
    -ts: Proportion of the dataset to be included in the test set.
    -bs: Number of samples per gradient update.
    -dr: Dropout rates per layer.
    -r: Indicator for regularization usage (0 - no regularization,
        1 - dropout method).
    -es: Indicator for early stopping usage (0 - no early stopping,
        1 - using early stopping).
    -a: The name of the activation function that will be used.
# Returns
    None. Saves the model and the corresponding training data for the model.
"""

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import plot_model
from nn_model import NNModel
import argparse
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import pandas as pd


def set_parser_arguments(parser):
    """Function that sets argument parser's options for command line
    interface invocation.

    # Returns
        ArgumentParser object
    """
    # Path to the image
    parser.add_argument(
        "-ip",
        "--image_path",
        required=False,
        default="../resources/images/",
        help="path to input images",
    )

    # Path to the parameters of the images
    parser.add_argument(
        "-pp",
        "--par_path",
        required=False,
        default="../resources/images/params/",
        help="path to input parameters of images (R, H)",
    )

    # Number of epochs for training
    parser.add_argument(
        "-e",
        "--epochs",
        required=False,
        default=15,
        help="number of epochs for training",
    )

    # Learning rate
    parser.add_argument(
        "-lr",
        "--learning_rate",
        required=False,
        default=0.00001,
        help="learning rate for optimizer",
    )

    # Number of repeated training of the same module architecture
    parser.add_argument(
        "-i",
        "--iterations",
        required=False,
        default=8,
        help="number of training iteration for the same " "architecture",
    )

    # List of number of units per hidden layer
    parser.add_argument(
        "-hl",
        "--hidden_layers",
        required=False,
        nargs="+",
        default=[108, 55],
        help="number of units per hidden layer",
    )

    # The directory for the outputs of the model.
    parser.add_argument(
        "-mo",
        "--model_output",
        required=False,
        default="../resources/model_output/",
        help="path for model outputs",
    )

    # Proportion of the dataset to be included in the test set.
    parser.add_argument(
        "-ts",
        "--test_split",
        required=False,
        default=0.2,
        help="proportion of the dataset to include in the " "test",
    )

    # Number of samples per gradient update.
    parser.add_argument(
        "-bs",
        "--batch_size",
        required=False,
        default=32,
        help="number of samples per gradient update",
    )

    # Dropout rates per layer.
    parser.add_argument(
        "-dr",
        "--dropout_rates",
        required=False,
        nargs="+",
        default=[0.3, 0.1],
        help="droupout rates",
    )

    # Indicator for regularization usage
    parser.add_argument(
        "-r",
        "--regularization",
        required=False,
        default=0,
        help="regularization (0 - no regularization, 1 - " "dropout method)",
    )

    # Indicator for early stopping usage
    parser.add_argument(
        "-es",
        "--early_stopping",
        required=False,
        default=1,
        help="early stopping (0 - no early stopping, 1 - " "using early stopping)",
    )

    # Activation function
    parser.add_argument(
        "-a",
        "--activation_func",
        required=False,
        default="relu",
        help="the name of the activation function" " (relu, tanh)",
    )


def load_data(dir_imgs, dir_params):
    """Load the data and parameters from the given directory
    location of the files.

    # Args
        dir_imgs: Path to the directory of the images.
        dir_params: Path to the directory of parameters.

    # Returns
        Tuple of data and parameters.
    """
    data = []
    params = []

    for i in range(0, 1000):
        img_path = dir_imgs + "SLIKA" + str(i + 1) + ".png"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        data.append(image)
        param_path = dir_params + "input" + str(i + 1) + "/INPUT.TXT"
        params.append(np.loadtxt(param_path))

    data = np.array(data)
    params = np.array(params)

    return data, params


def create_features(data):
    """Function that creates features for the model from the image.

    # Args
        data: Image (observation, height, width)

    # Returns
        Pixels on the radius of the images.
    """
    data_features = data[:, 100, 100:201]
    return data_features


def scale_features(data, params, model_out):
    """Scaling features of input and output variables between 0 and 1.

    # Args
        data: Input variables.
        params: Output variables.
        model_out: Location where the scalers will be saved.

    # Returns
        Tuple of scaled input and output variables.
    """
    # Scaling the features and output to the range between 0 and 1
    sc = MinMaxScaler(feature_range=(0, 1))

    data = sc.fit_transform(data)
    joblib.dump(sc, "{}data.scaler".format(model_out))
    params = sc.fit_transform(params)
    joblib.dump(sc, "{}params.scaler".format(model_out))

    return data, params


class Model:
    def __init__(
        self, regularization, h_layers, learning_rate, activation_func, dropout_rates
    ):
        self.regularization = regularization
        self.h_layers = h_layers
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.dropout_rates = dropout_rates

    def _create_model(
        self, regularization, h_layers, learning_rate, activation_func, dropout_rates
    ):
        """Function that creates and returns keras model.

        # Args
            regularization: Indicator whether to use regularization
                (0 - no regularization, 1 - dropout method).
            h_layers: List of units per hidden layer
            learning_rate: Learning rate of the NN.
            activation_func: Activation function used in hidden layer units.
            dropout_rates: List of dropout rates per hidden layer.

        # Returns
            Keras model.
        """
        # Building the keras model with or without dropout regularization
        if regularization == 0:
            model = NNModel.build(h_layers, 101, learning_rate, activation_func)
        elif regularization == 1:
            model = NNModel.build_dropout(
                h_layers, 101, learning_rate, dropout_rates, activation_func
            )
        else:
            raise ValueError(
                f"""Wrong regularization value. Regularization
                                     value cannot be {regularization}"""
            )

        return model

    def create_instance(self):
        """Creates model with randomly initialized weights with the parameters
        already given in the constructor

        # Returns
            Model with randomly initialized weights.
        """
        return self._create_model(
            self.regularization,
            self.h_layers,
            self.learning_rate,
            self.activation_func,
            self.dropout_rates,
        )


class SessionStatistic:
    def __init__(self):
        # Mean absolute errors for both
        self.mae_test = []
        self.mae_training = []

        # Root mean square error for both
        self.rmse_test = []
        self.rmse_training = []

        # Mean absolute errors for radius and depth
        self.mae_r_test = []
        self.mae_h_test = []
        self.mae_r_training = []
        self.mae_h_training = []

        # Root mean square errors for radius and depth
        self.rmse_r_test = []
        self.rmse_h_test = []
        self.rmse_r_training = []
        self.rmse_h_training = []

        # Running times
        self.times = []

        # Standard deviation of multiple training iterations for mean absolute error
        self.std_mae_test = None
        self.std_mae_training = None

        # Standard deviation of multiple training iterations for RMSE
        self.std_rmse_test = None
        self.std_rmse_training = None

        # Standard deviation of MAEs
        self.std_mae_r_test = None
        self.std_mae_h_test = None
        self.std_mae_r_training = None
        self.std_mae_h_training = None

        # Standard deviation of RMSEs
        self.std_rmse_r_test = None
        self.std_rmse_h_test = None
        self.std_rmse_r_training = None
        self.std_rmse_h_training = None

        # Mean of multiple training iterations for mean absolute error
        self.mean_mae_test = None
        self.mean_mae_training = None

        # Mean of multiple training iterations for RMSE
        self.mean_rmse_test = None
        self.mean_rmse_training = None

        # Mean of MAEs
        self.mean_mae_r_test = None
        self.mean_mae_h_test = None
        self.mean_mae_r_training = None
        self.mean_mae_h_training = None

        # Mean of RMSEs
        self.mean_rmse_r_test = None
        self.mean_rmse_h_test = None
        self.mean_rmse_r_training = None
        self.mean_rmse_h_training = None

    def calculate_stds(self):
        """Calculating the standard deviation for the errors"""

        self.std_mae_test = np.std(self.mae_test)
        self.std_mae_training = np.std(self.mae_training)

        self.std_rmse_test = np.std(self.rmse_test)
        self.std_rmse_training = np.std(self.rmse_training)

        self.std_mae_r_test = np.std(self.mae_r_test)
        self.std_mae_h_test = np.std(self.mae_h_test)
        self.std_mae_r_training = np.std(self.mae_r_training)
        self.std_mae_h_training = np.std(self.mae_h_training)

        self.std_rmse_r_test = np.std(self.rmse_r_test)
        self.std_rmse_h_test = np.std(self.rmse_h_test)
        self.std_rmse_r_training = np.std(self.rmse_r_training)
        self.std_rmse_h_training = np.std(self.rmse_h_training)

    def calculate_means(self):
        """Calculating the mean value for the errors"""
        self.mean_mae_test = np.mean(self.mae_test)
        self.mean_mae_training = np.mean(self.mae_training)

        self.mean_rmse_test = np.mean(self.rmse_test)
        self.mean_rmse_training = np.mean(self.rmse_training)

        self.mean_mae_r_test = np.mean(self.mae_r_test)
        self.mean_mae_h_test = np.mean(self.mae_h_test)
        self.mean_mae_r_training = np.mean(self.mae_r_training)
        self.mean_mae_h_training = np.mean(self.mae_h_training)

        self.mean_rmse_r_test = np.mean(self.rmse_r_test)
        self.mean_rmse_h_test = np.mean(self.rmse_h_test)
        self.mean_rmse_r_training = np.mean(self.rmse_r_training)
        self.mean_rmse_h_training = np.mean(self.rmse_h_training)

    def write_statistics_to_file(self, directory_path):
        """Creating CSV file with mean absolute errors and root mean
        squared errors for every iteration.

        # Params
            directory_path: Directory path for model outputs
        """
        errors_arr = np.vstack(
            (
                self.mae_training,
                self.mae_test,
                self.rmse_training,
                self.rmse_test,
                self.mae_r_training,
                self.mae_r_test,
                self.mae_h_training,
                self.mae_h_test,
                self.rmse_r_training,
                self.rmse_r_test,
                self.rmse_h_training,
                self.rmse_h_test,
            )
        )
        errors_arr = errors_arr.T
        errors_df = pd.DataFrame(errors_arr)
        errors_df.columns = [
            "MAE training",
            "MAE test",
            "RMSE training",
            "RMSE test",
            "MAE R training",
            "MAE R test",
            "MAE H training",
            "MAE H test",
            "RMSE R training",
            "RMSE R test",
            "RMSE H training",
            "RMSE H test",
        ]
        errors_df.to_csv(os.path.join(directory_path, "errors.csv"))

    def write_mae_statistic_to_file(self, directory_path):
        """Create a CSV file for MAEs.

        # Params
            directory_path: Directory path for model outputs
        """
        r_list = [
            self.mean_mae_r_training,
            self.mean_mae_r_test,
            self.std_mae_r_training,
            self.std_mae_r_test,
        ]
        h_list = [
            self.mean_mae_h_training,
            self.mean_mae_h_test,
            self.std_mae_h_training,
            self.std_mae_h_test,
        ]
        both_list = [
            self.mean_mae_training,
            self.mean_mae_test,
            self.std_mae_training,
            self.std_mae_test,
        ]

        maes_df = pd.DataFrame(np.array((r_list, h_list, both_list)))
        maes_df.columns = ["mean training", "mean test", "std training", "std test"]
        maes_df.rename(index={0: "radius", 1: "depth", 2: "both"}, inplace=True)
        maes_df.to_csv(os.path.join(directory_path, "maes.csv"))

    def write_rmse_statistic_to_file(self, directory_path):
        """Creating a CSV file for RMSEs.

        # Params
            directory_path: Directory path for model outputs
        """
        r_list = [
            self.mean_rmse_r_training,
            self.mean_rmse_r_test,
            self.std_rmse_r_training,
            self.std_rmse_r_test,
        ]
        h_list = [
            self.mean_rmse_h_training,
            self.mean_rmse_h_test,
            self.std_rmse_h_training,
            self.std_rmse_h_test,
        ]
        both_list = [
            self.mean_rmse_training,
            self.mean_rmse_test,
            self.std_rmse_training,
            self.std_rmse_test,
        ]

        rmaes_df = pd.DataFrame(np.array((r_list, h_list, both_list)))
        rmaes_df.columns = ["mean training", "mean test", "std training", "std test"]
        rmaes_df.rename(index={0: "radius", 1: "depth", 2: "both"}, inplace=True)
        rmaes_df.to_csv(os.path.join(directory_path, "rmses.csv"))

    def write_statistic_human_friendly_format_to_file(self, directory_path):
        """Printing descriptive statistics in a human friendly format.

        # Params
            directory_path: Directory path for model outputs
        """
        # Print mean absolute errors
        file_stat = open(os.path.join(directory_path, "stats.txt"), "w")
        file_stat.write(
            "MAE_train = {}\t STD_train = {}\n".format(
                self.mean_mae_training, self.std_mae_training
            )
        )
        file_stat.write(
            "MAE_test = {}\t STD_test = {}\n".format(
                self.mean_mae_test, self.std_mae_test
            )
        )
        file_stat.write("-----------------------------------\n")
        file_stat.write(
            "MAE_train (R) = {}\t STD_train (R) = {}\n".format(
                self.mean_mae_r_training, self.std_mae_r_training
            )
        )
        file_stat.write(
            "MAE_train (H) = {}\t STD_train (H) = {}\n".format(
                self.mean_mae_h_training, self.std_mae_h_training
            )
        )
        file_stat.write(
            "MAE_test (R) = {}\t STD_test (R) = {}\n".format(
                self.mean_mae_r_test, self.std_mae_r_test
            )
        )
        file_stat.write(
            "MAE_test (H) = {}\t STD_test (H) = {}\n".format(
                self.mean_mae_h_test, self.std_mae_h_test
            )
        )

        file_stat.write("\n************************************\n\n")

        # Print root mean square errors
        file_stat.write(
            "RMSE_train = {}  STD_train = {}\n".format(
                self.mean_rmse_training, self.std_rmse_training
            )
        )
        file_stat.write(
            "RMSE_test = {}  STD_test = {}\n".format(
                self.mean_rmse_test, self.std_rmse_test
            )
        )
        file_stat.write("-----------------------------------\n")
        file_stat.write(
            "RMSE_train (R) = {}\t STD_train (R) = {}\n".format(
                self.mean_rmse_r_training, self.std_rmse_r_training
            )
        )
        file_stat.write(
            "RMSE_train (H) = {}\t STD_train (H) = {}\n".format(
                self.mean_rmse_h_training, self.std_rmse_h_training
            )
        )
        file_stat.write(
            "RMSE_test (R) = {}\t STD_test (R) = {}\n".format(
                self.mean_rmse_r_test, self.std_rmse_r_test
            )
        )
        file_stat.write(
            "RMSE_test (H) = {}\t STD_test (H) = {}\n".format(
                self.mean_rmse_h_test, self.std_rmse_h_test
            )
        )

        file_stat.close()

    def write_running_times_to_file(self, directory_path):
        """Write running times to a file.

        # Params
            directory_path: Directory path for model outputs
        """
        times = np.array(self.times)
        running_times_df = pd.DataFrame(times.T)
        running_times_df.columns = ["running times"]
        running_times_df.to_csv(os.path.join(directory_path, "running_times.csv"))


def _plot_loss_function(loss, val_loss, iteration, directory_path):
    """Plot loss function.

    # Args
        loss: List of loss values on the training set
        val_loss: List of loss values on the validation set
        iteration: Iteration number
        directory_path: Directory path for model outputs
    """
    epochs_list = np.arange(0, len(loss))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epochs_list, loss, label="loss_training")
    plt.plot(epochs_list, val_loss, label="loss_cv")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss function (training, cv)")
    plt.legend()
    plt.savefig(os.path.join(directory_path, f"iter_{iteration}", "loss_training.png"))
    plt.close()


def _plot_metric_function(metric, metric_val, iteration, directory_path):
    """Plotting metrics per epoch.

    # Args
        metric: List of metric values on the training set
        metric_val: List of metric values on the validation set
        iteration: Iteration number
        directory_path: Directory path for model outputs
    """
    epochs_list = np.arange(0, len(metric))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epochs_list, metric, label="mean_absolute_error_train")
    plt.plot(epochs_list, metric_val, label="mean_absolute_error_cv")
    plt.title("Mean absolute error (training, cv)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.legend()
    plt.savefig(
        os.path.join(directory_path, f"iter_{iteration}", "metric_training.png")
    )
    plt.close()


def _plot_prediction_scatter_plot(
    x,
    y,
    xlabel,
    ylabel,
    title,
    iteration,
    directory_path,
    file_output,
    plt_low,
    plt_high,
):
    """Plotting the scatter plot of predictions for given parameters.

    # Args
        x: x-axis values
        y: y-axis values
        xlabel: x-axis label
        ylabel: y-axis label
        title: Title of the plot
        iteration: Iteration number
        directory_path: Directory path for model outputs
        file_output: File name of the plot
        plt_low: Lowest value to be shown on the plot
        plt_high: Highest value to be shown on the plot
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(x, y, "b.", markersize=1)
    plt.plot([plt_low, plt_high], [plt_low, plt_high], "r")
    plt.xlim((plt_low - 500, plt_high + 500))
    plt.ylim((plt_low - 500, plt_high + 500))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(directory_path, f"iter_{iteration}", file_output))
    plt.close()


def train(
    model,
    data,
    params,
    test_split,
    batch_size,
    epochs,
    iterations,
    ind_early_stopping,
    model_out,
):
    """Trains the model.

    # Args
        model: Model class object that can create instance of the
            keras model
        data: Training data
        params: Labels
        test_split: Percentage of the data used for test set
        batch_size: Number of samples per gradient update
        epochs: Number of epochs for training
        iterations: Number of training iterations for the same architecture
        ind_early_stopping: Indicator for early stopping usage
        model_out: The directory for the outputs of the model

    # Returns
        Trained model
    """
    keras_model = model.create_instance()

    print("[INFO] printing model summary")
    keras_model.summary()

    session_statistic = SessionStatistic()

    earlystop = EarlyStopping(
        monitor="val_mean_absolute_error",
        min_delta=0.0001,
        patience=25,
        verbose=1,
        mode="auto",
    )

    callback_list = []
    if ind_early_stopping:
        callback_list.append(earlystop)

    for i in range(0, iterations):
        t_start = time.time()
        print("ITERATION #{}".format(i))
        print("[INFO] spliting data")
        train_X, test_X, train_y, test_y = train_test_split(
            data, params, test_size=test_split
        )

        keras_model = model.create_instance()

        print("[INFO] training model")
        H = keras_model.fit(
            train_X,
            train_y,
            validation_split=0.25,
            epochs=epochs,
            verbose=1,
            batch_size=batch_size,
            callbacks=callback_list,
        )

        # Predicting the R and H with the model
        predicted_test = keras_model.predict(test_X)
        predicted_training = keras_model.predict(train_X)

        # Loading the normalizer and transforming the values back to
        # normal range
        sc = joblib.load("{}params.scaler".format(model_out))
        predicted_test_scaled = sc.inverse_transform(predicted_test)
        predicted_training_scaled = sc.inverse_transform(predicted_training)
        testY_scaled = sc.inverse_transform(test_y)
        trainY_scaled = sc.inverse_transform(train_y)

        # Calculating the mean absolute error
        current_mae_test = mean_absolute_error(testY_scaled, predicted_test_scaled)
        current_mae_training = mean_absolute_error(
            trainY_scaled, predicted_training_scaled
        )

        # Calculating the root mean squared error
        current_rmse_test = sqrt(
            mean_squared_error(testY_scaled, predicted_test_scaled)
        )
        current_rmse_training = sqrt(
            mean_squared_error(trainY_scaled, predicted_training_scaled)
        )

        # Appending mean absolute errors to the session statistic
        session_statistic.mae_test.append(current_mae_test)
        session_statistic.mae_training.append(current_mae_training)

        # Appending root mean square error to session statistic
        session_statistic.rmse_test.append(current_rmse_test)
        session_statistic.rmse_training.append(current_rmse_training)

        # Calculating the mean absolute error of R and H for test and training set
        session_statistic.mae_r_test.append(
            mean_absolute_error(testY_scaled[:, 0], predicted_test_scaled[:, 0])
        )
        session_statistic.mae_h_test.append(
            mean_absolute_error(testY_scaled[:, 1], predicted_test_scaled[:, 1])
        )
        session_statistic.mae_r_training.append(
            mean_absolute_error(trainY_scaled[:, 0], predicted_training_scaled[:, 0])
        )
        session_statistic.mae_h_training.append(
            mean_absolute_error(trainY_scaled[:, 1], predicted_training_scaled[:, 1])
        )

        # Calculating the root mean squared error of R and H for test and training set
        session_statistic.rmse_r_test.append(
            sqrt(mean_squared_error(testY_scaled[:, 0], predicted_test_scaled[:, 0]))
        )
        session_statistic.rmse_h_test.append(
            sqrt(mean_squared_error(testY_scaled[:, 1], predicted_test_scaled[:, 1]))
        )
        session_statistic.rmse_r_training.append(
            sqrt(
                mean_squared_error(trainY_scaled[:, 0], predicted_training_scaled[:, 0])
            )
        )
        session_statistic.rmse_h_training.append(
            sqrt(
                mean_squared_error(trainY_scaled[:, 1], predicted_training_scaled[:, 1])
            )
        )

        t_end = time.time()
        session_statistic.times.append(t_end - t_start)

        # Create folder for the current iteration if it doesn't exist
        if not os.path.exists(os.path.join(model_out, f"iter_{i}")):
            os.makedirs(os.path.join(model_out, f"iter_{i}"))

        # Saving the model
        keras_model.save(os.path.join(model_out, f"iter_{i}", "model.model"))

        # Ploting the training plot for the last iteration
        print("[INFO] ploting mean absolute error")
        _plot_metric_function(
            H.history["mean_absolute_error"],
            H.history["val_mean_absolute_error"],
            i,
            model_out,
        )

        # Saving the predictions of the model
        predictions = np.hstack((predicted_test_scaled, testY_scaled))
        np.savetxt(
            os.path.join(model_out, f"iter_{i}", "predictions.csv"),
            predictions,
            delimiter=",",
            fmt="%.2f",
        )

        print("[INFO] ploting scatter plot of predictions")
        # Ploting the scatter plot of predictions for R
        _plot_prediction_scatter_plot(
            x=testY_scaled[:, 0],
            y=predictions[:, 0],
            xlabel="Real R",
            ylabel="Predicted R",
            title="Predicted vs Real R",
            iteration=i,
            directory_path=model_out,
            file_output="pred_vs_realR.png",
            plt_low=1000,
            plt_high=5000,
        )

        # Ploting the scatter plot of prediction for H
        _plot_prediction_scatter_plot(
            x=testY_scaled[:, 1],
            y=predictions[:, 1],
            xlabel="Real H",
            ylabel="Predicted H",
            title="Predicted vs Real H",
            iteration=i,
            directory_path=model_out,
            file_output="pred_vs_realH.png",
            plt_low=100,
            plt_high=10000,
        )

        print("[INFO] ploting loss")
        # Ploting the loss function
        _plot_loss_function(H.history["loss"], H.history["val_loss"], i, model_out)

    # Calculating the mean value for the errors
    session_statistic.calculate_means()

    # Calculating the standard deviation for the errors
    session_statistic.calculate_stds()

    print("[INFO] printing descriptive statistics")
    # Creating CSV file with mean absolute errors and root mean squared erros for every iteration
    session_statistic.write_statistics_to_file(model_out)

    # Creating a CSV file for MAEs
    session_statistic.write_mae_statistic_to_file(model_out)

    # Creating a CSV file for RMSEs
    session_statistic.write_rmse_statistic_to_file(model_out)

    # Printing descriptive statistics in a human friendly format
    session_statistic.write_statistic_human_friendly_format_to_file(model_out)

    print("[INFO] printing model architecture")

    # Printing the architecture of the model
    plot_model(
        keras_model,
        to_file=os.path.join(model_out, "model.png"),
        show_layer_names=True,
        show_shapes=True,
    )

    # Printing running times
    session_statistic.write_running_times_to_file(model_out)


def run_train(args):
    # Getting directory paths for images, parameters and model output
    dir_imgs = args["image_path"]
    dir_params = args["par_path"]
    model_out = args["model_output"]

    print("[INFO] loading images")
    data, params = load_data(dir_imgs, dir_params)

    print("[INFO] creating features")
    # Creating features for a model
    data = create_features(data)

    print("[INFO] scaling features")
    data, params = scale_features(data, params, model_out)

    # Getting parameters for creating model
    regularization = int(args["regularization"])
    h_layers = np.array([int(x) for x in args["hidden_layers"]])
    learning_rate = float(args["learning_rate"])
    activation_func = args["activation_func"]
    dropout_rates = np.array([float(x) for x in args["dropout_rates"]])

    print("[INFO] creating model architecture")
    model_class = Model(
        regularization=regularization,
        h_layers=h_layers,
        learning_rate=learning_rate,
        activation_func=activation_func,
        dropout_rates=dropout_rates,
    )

    # Getting parameters for model training
    ind_early_stopping = bool(int(args["early_stopping"]))
    test_split = float(args["test_split"])
    batch_size = int(args["batch_size"])
    epochs = int(args["epochs"])
    iterations = int(args["iterations"])

    train(
        model=model_class,
        data=data,
        params=params,
        test_split=test_split,
        batch_size=batch_size,
        epochs=epochs,
        iterations=iterations,
        ind_early_stopping=ind_early_stopping,
        model_out=model_out,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    set_parser_arguments(ap)
    arguments = vars(ap.parse_args())

    run_train(arguments)
