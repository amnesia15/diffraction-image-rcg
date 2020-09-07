import context
import numpy as np
from keras.models import Sequential

from diffraction_image_rcg.nn_model import NNModel


def test_generate_combination_low_high():
    depth = 3
    low_no = 5
    high_no = 7
    values = NNModel.generate_combination_low_high(depth, low_no, high_no)

    expected_values = np.asarray(
        [
            [5, 5, 5],
            [5, 5, 6],
            [5, 5, 7],
            [5, 6, 5],
            [5, 6, 6],
            [5, 6, 7],
            [5, 7, 5],
            [5, 7, 6],
            [5, 7, 7],
            [6, 5, 5],
            [6, 5, 6],
            [6, 5, 7],
            [6, 6, 5],
            [6, 6, 6],
            [6, 6, 7],
            [6, 7, 5],
            [6, 7, 6],
            [6, 7, 7],
            [7, 5, 5],
            [7, 5, 6],
            [7, 5, 7],
            [7, 6, 5],
            [7, 6, 6],
            [7, 6, 7],
            [7, 7, 5],
            [7, 7, 6],
            [7, 7, 7],
        ]
    )

    assert np.array_equal(values, expected_values)


def test_generate_combination_low_high_different():
    low_high_arr = np.asarray([[2, 5], [7, 8]])

    values = NNModel.generate_combination_low_high_different(low_high_arr)

    expected_values = np.asarray(
        [[2, 7], [2, 8], [3, 7], [3, 8], [4, 7], [4, 8], [5, 7], [5, 8]]
    )

    assert np.array_equal(values, expected_values)


def test_nn_model_build():
    # Arrange
    hidden_units = np.asarray([10, 10])
    input_dimension = 16
    learning_rate = 0.005
    activation_fun = "relu"

    # Act
    model = NNModel.build(
        hidden_units=hidden_units,
        input_dimension=input_dimension,
        learning_rate=learning_rate,
        activation_func=activation_fun,
    )

    assert isinstance(model, Sequential)


def test_nn_model_build_dropout():
    # Arrange
    hidden_units = np.asarray([10, 10])
    input_dimension = 16
    learning_rate = 0.005
    activation_fun = "relu"
    dropout_rate = np.asarray([10, 10])

    # Act
    model = NNModel.build_dropout(
        hidden_units=hidden_units,
        input_dimension=input_dimension,
        learning_rate=learning_rate,
        activation_func=activation_fun,
        dropout_rate=dropout_rate,
    )

    assert isinstance(model, Sequential)
