import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_contour_plot(x, y, z, save_path):
    """Function interpolates contour lines and then saves the
    figure of the contour plot. Countour plot represents mean
    light intensity of the diffraction image for the given
    parameters of radius and depth.

    # Args
        x: x-axis coordinates of the radius parameter.
        y: y-axis coordinates of the depth parameter.
        z: Mean values for the given radius and depth parameter.
        save_path: path where the plot figure will be saved.

    # Returns
        None.
    """
    plt.contourf(x, y, z, 100)
    plt.xlabel("R (nm)")
    plt.ylabel("H (nm)")
    plt.colorbar()
    plt.title("Contour plot (mean value)")
    plt.savefig(save_path)


def plot_running_times(
    run_relu_1,
    run_tanh_1,
    run_relu_2,
    run_tanh_2,
    run_relu_3,
    run_tanh_3,
    run_relu_4,
    run_tanh_4,
    save_path,
):
    """
    Functions plots the training execution times for various neural
    network architectures.

    # Args
        run_relu_1: running times of the trainings for neural network
                    with hidden layer units: 108, 55
        run_tanh_1: running times of the trainings for neural network
                    with hidden layer units: 108, 55
        run_relu_2: running times of the trainings for neural network
                    with hidden layer units: 200, 50, 25
        run_tanh_2: running times of the trainings for neural network
                    with hidden layer units: 200, 50, 25
        run_relu_3: running times of the trainings for neural network
                    with hidden layer units: 202, 101, 50, 20
        run_tanh_3: running times of the trainings for neural network
                    with hidden layer units: 202, 101, 50, 20
        run_relu_4: running times of the trainings for neural network
                    with hidden layer units: 2048, 1024, 1024, 512, 512, 256
        run_tanh_4: running times of the trainings for neural network
                    with hidden layer units: 2048, 1024, 1024, 512, 512, 256
        save_path: path where the plot figure will be saved.

    # Returns
        None.
    """
    labels = ["108-55", "200-50-25", "202-101-50-20", "2048-1024-1024-512-512-256"]

    labelsize = 7
    mpl.rcParams["xtick.labelsize"] = labelsize
    mpl.rcParams["ytick.labelsize"] = labelsize

    # Calculating means of running times for relu NNs and tanh NNs
    n_groups = 4
    means_running_relu = [
        run_relu_1.mean(),
        run_relu_2.mean(),
        run_relu_3.mean(),
        run_relu_4.mean(),
    ]
    means_running_tanh = [
        run_tanh_1.mean(),
        run_tanh_2.mean(),
        run_tanh_3.mean(),
        run_tanh_4.mean(),
    ]

    # Creating bar plot of running times
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    # Plot Relu bars
    rects1 = plt.bar(
        index, means_running_relu, bar_width, alpha=opacity, color="b", label="Relu"
    )

    # Plot Tanh bars
    rects2 = plt.bar(
        index + bar_width,
        means_running_tanh,
        bar_width,
        alpha=opacity,
        color="g",
        label="Tanh",
    )

    plt.xlabel("Architectures")
    plt.ylabel("Seconds (mean)")
    plt.title("Running times")
    plt.xticks(index + bar_width, labels)
    plt.legend()

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")


def plot_mean_absolute_errors_barplot(
    df1_relu,
    df2_relu,
    df3_relu,
    df4_relu,
    df1_tanh,
    df2_tanh,
    df3_tanh,
    df4_tanh,
    save_path,
):
    """
    # Args
        df1_relu: pandas dataframe that contains information about
                  mean absolute error on the test set for neural network
                  with hidden layer units: 200, 50, 25 and relu activation
                  function.
        df2_relu: pandas dataframe that contains information about
                  mean absolute error on the test set for neural network
                  with hidden layer units: 200, 50, 25 and relu activation
                  function.
        df3_relu: pandas dataframe that contains information about
                  mean absolute error on the test set for neural network
                  with hidden layer units: 200, 50, 25 and relu activation
                  function.
        df4_relu: pandas dataframe that contains information about
                  mean absolute error on the test set for neural network
                  with hidden layer units: 200, 50, 25 and relu activation
                  function.
        df1_tanh: pandas dataframe that contains information about
                  mean absolute error on the test set for neural network
                  with hidden layer units: 108, 55 and tanh activation
                  function.
        df2_tanh: pandas dataframe that contains information about
                  mean absolute error on the test set for neural network
                  with hidden layer units: 200, 50, 25 and tanh activation
                  function.
        df3_tanh: pandas dataframe that contains information about
                  mean absolute error on the test set for neural network
                  with hidden layer units: 202, 101, 50, 20 and tanh
                  activation function.
        df4_tanh: pandas dataframe that contains information about
                  mean absolute error on the test set for neural network
                  with hidden layer units: 2048, 1024, 1024, 512, 512, 256
                  and tanh activation function.
        save_path: path where the plot figure will be saved.

    # Returns
        None.
    """
    labels = ["108-55", "200-50-25", "202-101-50-20", "2048-1024-1024-512-512-256"]

    # Calculating means of MAE test for relu NNs and tanh NNs
    n_groups = 4
    means_relu_mae = [
        df1_relu["MAE test"].mean(),
        df2_relu["MAE test"].mean(),
        df3_relu["MAE test"].mean(),
        df4_relu["MAE test"].mean(),
    ]
    means_tanh_mae = [
        df1_tanh["MAE test"].mean(),
        df2_tanh["MAE test"].mean(),
        df3_tanh["MAE test"].mean(),
        df4_tanh["MAE test"].mean(),
    ]

    # Creating bar plot of MAE TEST erros
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(
        index, means_relu_mae, bar_width, alpha=opacity, color="b", label="Relu"
    )

    rects2 = plt.bar(
        index + bar_width,
        means_tanh_mae,
        bar_width,
        alpha=opacity,
        color="g",
        label="Tanh",
    )

    plt.xlabel("Architectures")
    plt.ylabel("MAE")
    plt.title("Mean absolute errors (test)")
    plt.xticks(index + bar_width, labels)
    plt.legend()

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")


def load_data_contour_plot():
    # Read X values (radius)
    x = []
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "data", "parameter_space", "x_value.txt"
        )
    )
    f = open(path, "r")
    lines = f.readlines()
    f.close()

    for num in lines:
        x.append(int(num))

    # Read Y values (depth)
    y = []
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "data", "parameter_space", "y_value.txt"
        )
    )
    f = open(path, "r")
    lines = f.readlines()
    f.close()

    for num in lines:
        y.append(int(num))

    # Read the mean intensity for the given X and Y
    z = []
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "data", "parameter_space", "mean_matrix.txt"
        )
    )
    f = open(path, "r")
    for i in range(0, 10):
        new_row = []
        lines = f.readline().split(" ")
        for num in lines:
            new_row.append(float(num))
        z.append(new_row)
    f.close()

    return x, y, z


def load_data_running_times():
    # Loading running times of models with different activation
    # functions and architecture
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors" "108-55 (relu)",
            "model_output" "running_times.csv",
        )
    )
    run_relu_1 = pd.read_csv(path)["running times"]

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors" "108-55 (tanh)",
            "model_output" "running_times.csv",
        )
    )
    run_tanh_1 = pd.read_csv(path)["running times"]

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors" "200-50-25 (relu)",
            "model_output" "running_times.csv",
        )
    )
    run_relu_2 = pd.read_csv(path)["running times"]

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors" "200-50-25 (tanh)",
            "model_output" "running_times.csv",
        )
    )
    run_tanh_2 = pd.read_csv(path)["running times"]

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors" "202-101-50-20 (relu)",
            "model_output" "running_times.csv",
        )
    )
    run_relu_3 = pd.read_csv(path)["running times"]

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors" "202-101-50-20 (tanh)",
            "model_output" "running_times.csv",
        )
    )
    run_tanh_3 = pd.read_csv(path)["running times"]

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors" "2048-1024-1024-512-512-256 (relu)",
            "model_output" "running_times.csv",
        )
    )
    run_relu_4 = pd.read_csv(path)["running times"]

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors" "2048-1024-1024-512-512-256 (tanh)",
            "model_output" "running_times.csv",
        )
    )
    run_tanh_4 = pd.read_csv(path)["running times"]

    return (
        run_relu_1,
        run_tanh_1,
        run_relu_2,
        run_tanh_2,
        run_relu_3,
        run_tanh_3,
        run_relu_4,
        run_tanh_4,
    )


def load_errors():
    # Load the data about errors of the models for various architectures
    # (tanh and relu)
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors",
            "108-55 (relu)",
            "model_output",
            "errors.csv",
        )
    )
    df1_relu = pd.read_csv(path)

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors",
            "200-50-25 (relu)",
            "model_output",
            "errors.csv",
        )
    )
    df2_relu = pd.read_csv(path)

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors",
            "202-101-50-20 (relu)",
            "model_output",
            "errors.csv",
        )
    )
    df3_relu = pd.read_csv(path)

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors",
            "2048-1024-1024-512-512-256 (relu)",
            "model_output",
            "errors.csv",
        )
    )
    df4_relu = pd.read_csv(path)

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors",
            "108-55 (tanh)",
            "model_output",
            "errors.csv",
        )
    )
    df1_tanh = pd.read_csv(path)

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors",
            "200-50-25 (tanh)",
            "model_output",
            "errors.csv",
        )
    )
    df2_tanh = pd.read_csv(path)

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors",
            "202-101-50-20 (tanh)",
            "model_output",
            "errors.csv",
        )
    )
    df3_tanh = pd.read_csv(path)

    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "errors",
            "2048-1024-1024-512-512-256 (tanh)",
            "model_output",
            "errors.csv",
        )
    )
    df4_tanh = pd.read_csv(path)

    return (
        df1_relu,
        df2_relu,
        df3_relu,
        df4_relu,
        df1_tanh,
        df2_tanh,
        df3_tanh,
        df4_tanh,
    )


if __name__ == "__main__":
    # Create countour plot
    save_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "data", "parameter_space", "color_plot.png"
        )
    )
    x, y, z = load_data_contour_plot()
    plot_contour_plot(x, y, z, save_path)

    # Create barplot of running times
    save_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "data", "errors", "running_times_bar.png"
        )
    )
    args = load_data_running_times()
    plot_running_times(*args, save_path=save_path)
