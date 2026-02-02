import numpy as np
import pytest

from goph547lab01.gravity import (
    gravity_potential_point,
    gravity_effect_point,
)


def test_gravity_potential_simple():
    """
    Simple check: point mass directly below observation point.
    """
    x = np.array([0.0, 0.0, 0.0])
    xm = np.array([0.0, 0.0, -10.0])
    m = 1.0e7
    G = 6.674e-11

    r = 10.0
    expected = G * m / r

    result = gravity_potential_point(x, xm, m, G)

    assert np.isclose(result, expected)


def test_gravity_effect_simple():
    """
    Vertical gravity effect should be positive downward.
    """
    x = np.array([0.0, 0.0, 0.0])
    xm = np.array([0.0, 0.0, -10.0])
    m = 1.0e7
    G = 6.674e-11

    r = 10.0
    dz = 10.0
    expected = G * m * dz / (r**3)

    result = gravity_effect_point(x, xm, m, G)

    assert np.isclose(result, expected)


def test_gravity_decreases_with_distance():
    """
    Potential and gravity effect should decrease in magnitude
    as distance from the mass increases.
    """
    xm = np.array([0.0, 0.0, -10.0])
    m = 1.0e7

    x_near = np.array([0.0, 0.0, 0.0])
    x_far = np.array([0.0, 0.0, 100.0])

    U_near = gravity_potential_point(x_near, xm, m)
    U_far = gravity_potential_point(x_far, xm, m)

    gz_near = gravity_effect_point(x_near, xm, m)
    gz_far = gravity_effect_point(x_far, xm, m)

    assert abs(U_near) > abs(U_far)
    assert abs(gz_near) > abs(gz_far)
