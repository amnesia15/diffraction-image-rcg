import os
import context
import pytest
import pandas as pd
from visualization.plot import plot_contour_plot, \
    plot_running_times, plot_mean_absolute_errors_barplot


@pytest.fixture
def x_values():
    x = []
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'parameter_space',
                                        'x_value.txt'))
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    for num in lines:
        x.append(int(num))

    return x


@pytest.fixture
def y_values():
    y = []
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'parameter_space',
                                        'y_value.txt'))
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    for num in lines:
        y.append(int(num))

    return y


@pytest.fixture
def z_values():
    z = []
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'parameter_space',
                                        'mean_matrix.txt'))
    f = open(path, 'r')
    for i in range(0, 10):
        new_row = []
        lines = f.readline().split(' ')
        for num in lines:
            new_row.append(float(num))
        z.append(new_row)
    f.close()

    return z


@pytest.fixture
def run_relu_1():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'errors',
                                        '108-55 (relu)',
                                        'model_output',
                                        'running_times.csv'))
    df = pd.read_csv(path)['running times']
    return df


@pytest.fixture
def run_tanh_1():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'errors',
                                        '108-55 (tanh)',
                                        'model_output',
                                        'running_times.csv'))
    df = pd.read_csv(path)['running times']
    return df


@pytest.fixture
def run_relu_2():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'errors',
                                        '200-50-25 (relu)',
                                        'model_output',
                                        'running_times.csv'))
    df = pd.read_csv(path)['running times']
    return df


@pytest.fixture
def run_tanh_2():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'errors',
                                        '200-50-25 (tanh)',
                                        'model_output',
                                        'running_times.csv'))
    df = pd.read_csv(path)['running times']
    return df


@pytest.fixture
def run_relu_3():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'errors',
                                        '202-101-50-20 (relu)',
                                        'model_output',
                                        'running_times.csv'))
    df = pd.read_csv(path)['running times']
    return df


@pytest.fixture
def run_tanh_3():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'errors',
                                        '202-101-50-20 (tanh)',
                                        'model_output',
                                        'running_times.csv'))
    df = pd.read_csv(path)['running times']
    return df


@pytest.fixture
def run_relu_4():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'errors',
                                        '2048-1024-1024-512-512-256 (relu)',
                                        'model_output',
                                        'running_times.csv'))
    df = pd.read_csv(path)['running times']
    return df


@pytest.fixture
def run_tanh_4():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'visualization',
                                        'data',
                                        'errors',
                                        '2048-1024-1024-512-512-256 (tanh)',
                                        'model_output',
                                        'running_times.csv'))
    df = pd.read_csv(path)['running times']
    return df


def test_plot_contour_plot(x_values, y_values, z_values):
    path = 'color_plot.png'
    plot_contour_plot(x_values, y_values, z_values, path)
    assert os.path.isfile(path)
    os.remove(path)


def test_plot_running_times(run_relu_1,
                            run_tanh_1,
                            run_relu_2,
                            run_tanh_2,
                            run_relu_3,
                            run_tanh_3,
                            run_relu_4,
                            run_tanh_4):
    path = 'running_times_bar.png'
    plot_running_times(run_relu_1,
                       run_tanh_1,
                       run_relu_2,
                       run_tanh_2,
                       run_relu_3,
                       run_tanh_3,
                       run_relu_4,
                       run_tanh_4,
                       path)
    assert os.path.isfile(path)
    os.remove(path)
