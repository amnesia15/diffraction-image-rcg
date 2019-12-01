import os
import context
import pytest
import numpy as np
import cv2
from transform_image import add_padding
from finding_center import find_center, prepare_model_image


@pytest.fixture
def dummy_image():
    return np.ones((201, 201))


@pytest.fixture
def diffraction_image():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'images',
                                        'SLIKA1.png'))
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return image


@pytest.fixture
def diffraction_image_top_ext():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..',
                                        'images',
                                        'SLIKA1.png'))
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    augmented_image = add_padding(image=image,
                                  side='top',
                                  number=20)

    return augmented_image


def test_add_padding(dummy_image):
    augmented_image_top = add_padding(image=dummy_image,
                                      side='top',
                                      number=10)
    augmented_image_bottom = add_padding(image=dummy_image,
                                         side='bottom',
                                         number=15)
    augmented_image_right = add_padding(image=dummy_image,
                                        side='right',
                                        number=20)
    augmented_image_left = add_padding(image=dummy_image,
                                       side='left',
                                       number=25)

    assert np.all(augmented_image_top[0:10, :] == 0)
    assert np.array_equal(augmented_image_top[10:, :], dummy_image)

    assert np.all(augmented_image_bottom[-15:, :] == 0)
    assert np.array_equal(augmented_image_bottom[:-15, :], dummy_image)

    assert np.all(augmented_image_right[:, -20:] == 0)
    assert np.array_equal(augmented_image_right[:, :-20], dummy_image)

    assert np.all(augmented_image_left[:, 0:25] == 0)
    assert np.array_equal(augmented_image_left[:, 25:], dummy_image)


def test_find_center(diffraction_image):
    center_x, center_y = find_center(diffraction_image)

    assert center_x == 100
    assert center_y == 100


def test_find_center_top_ext(diffraction_image_top_ext):
    center_x, center_y = find_center(diffraction_image_top_ext)

    assert center_x == 100
    assert center_y == 120


def test_prepare_model_image(diffraction_image, diffraction_image_top_ext):
    image = prepare_model_image(diffraction_image_top_ext)

    assert np.array_equal(image, diffraction_image)
