import os
import context
import pytest
from visualization.plot import plot_contour_plot


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


def test_plot_contour_plot(x_values, y_values, z_values):
    path = 'color_plot.png'
    plot_contour_plot(x_values, y_values, z_values, path)
    assert os.path.isfile(path)
    os.remove(path)
