import context
import pytest
import numpy as np

from diffraction_image_rcg.train import load_data, create_features


def test_load_data_returns_data_shapes_correctly():
    # Arrange
    dir_imgs = "./resources/images/"
    dir_params = "./resources/images/params/"

    # Act
    data, params = load_data(dir_imgs, dir_params)

    assert data.shape == (1000, 201, 201) and params.shape == (1000, 2)


def test_create_features_returns_shapes_correctly():
    # Arrange
    dir_imgs = "./resources/images/"
    dir_params = "./resources/images/params/"
    data, _ = load_data(dir_imgs, dir_params)

    # Act
    data_features = create_features(data)

    # Assert
    assert data_features.shape == (1000, 101)
