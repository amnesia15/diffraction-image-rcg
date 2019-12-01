import context
import pytest
import numpy as np
from transform_image import add_padding


@pytest.fixture
def dummy_image():
    return np.ones((201, 201))


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
