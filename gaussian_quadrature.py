import numpy as np

def gaussian_quadrature(n, a=-1.0, b=1.0):
    """
    Returns the Gauss-Legendre quadrature points and weights.

    Parameters
    ----------
    n : int
        Number of Gauss points.
    a : float, optional
        Lower limit of the interval (default: -1).
    b : float, optional
        Upper limit of the interval (default: 1).

    Returns
    -------
    x : ndarray
        Gauss points.
    w : ndarray
        Gauss weights.
    """

    # Gauss points and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n)

    # Transform to [a, b] if necessary
    if a != -1.0 or b != 1.0:
        x = 0.5 * (b - a) * x + 0.5 * (a + b)
        w = 0.5 * (b - a) * w

    return x, w