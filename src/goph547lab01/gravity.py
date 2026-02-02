from __future__ import annotations

from typing import Sequence
import numpy as np


def gravity_potential_point(
    x: Sequence[float],
    xm: Sequence[float],
    m: float,
    G: float = 6.674e-11,
) -> float:
    """Compute the gravity potential due to a point mass.

    Parameters
    ----------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.674e-11
        Constant of gravitation. Default in SI units.

    Returns
    -------
    float
        Gravity potential at x due to anomaly at xm.
    """
    x = np.asarray(x, dtype=float).reshape(3)
    xm = np.asarray(xm, dtype=float).reshape(3)

    r_vec = x - xm
    r = np.linalg.norm(r_vec)
    if r == 0.0:
        raise ValueError("Survey point x cannot equal mass location xm (r = 0).")

    # U = G*m/r
    return float(G * m / r)


def gravity_effect_point(
    x: Sequence[float],
    xm: Sequence[float],
    m: float,
    G: float = 6.674e-11,
) -> float:
    """Compute the vertical gravity effect due to a point mass (positive downward).

    Parameters
    ----------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.674e-11
        Constant of gravitation. Default in SI units.

    Returns
    -------
    float
        Gravity effect at x due to anomaly at xm.
    """
    x = np.asarray(x, dtype=float).reshape(3)
    xm = np.asarray(xm, dtype=float).reshape(3)

    r_vec = x - xm
    r = np.linalg.norm(r_vec)
    if r == 0.0:
        raise ValueError("Survey point x cannot equal mass location xm (r = 0).")

    # gz (positive downward) = G*m*(z - zm) / r^3
    dz = x[2] - xm[2]
    return float(G * m * dz / (r**3))
