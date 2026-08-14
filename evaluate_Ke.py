import numpy as np


def evaluate_Ke(x, theta, a, b, range_z, range_theta):
    """
    Python version of evaluate_Ke.m

    Parameters
    ----------
    x : float
    theta : float
    a : float
    b : float
    range_z : array_like
    range_theta : array_like

    Returns
    -------
    I_Ke : ndarray
    """

    # ------------------------------------------------------------------
    # Penalty stiffnesses
    # ------------------------------------------------------------------
    Ku_top = 1e8
    Kv_top = 1e8
    Kw_top = 1e8
    Kbz_top = 1e8
    Kbt_top = 1e8

    Rc = b

    range_z = np.asarray(range_z)
    range_theta = np.asarray(range_theta)

    # ------------------------------------------------------------------
    # Shape functions evaluated at x = a
    # ------------------------------------------------------------------

    F = np.sin(range_z * np.pi)
    G = np.sin(range_theta * theta)

    H = np.sin(range_z * np.pi)
    I = np.cos(range_theta * theta)

    #NFGx = (range_z * np.pi / a) * np.cos(range_z * np.pi) * np.sin(range_theta * theta)
    #NFGy = F * range_theta * np.cos(range_theta * theta)

    #NHIx = (range_z * np.pi / a) * np.cos(range_z * np.pi) * np.cos(range_theta * theta)
    #NHIy = -H * range_theta * np.sin(range_theta * theta)

    # ------------------------------------------------------------------
    # MATLAB reshape(A.',1,[])
    # ------------------------------------------------------------------

    NFG = np.outer(G, F).reshape(-1)
    NHI = np.outer(I, H).reshape(-1)

    NFGx = np.outer(
        np.sin(range_theta * theta),
        (range_z * np.pi / a) * np.cos(range_z * np.pi)
    ).reshape(-1)

    NFGy = np.outer(
        range_theta * np.cos(range_theta * theta),
        F
    ).reshape(-1)

    NHIx = np.outer(
        np.cos(range_theta * theta),
        (range_z * np.pi / a) * np.cos(range_z * np.pi)
    ).reshape(-1)

    NHIy = np.outer(
        -range_theta * np.sin(range_theta * theta),
        H
    ).reshape(-1)

    # ------------------------------------------------------------------
    # Assemble vectors
    # ------------------------------------------------------------------

    Nx = np.concatenate((NFGx, NHIx))
    Ny = np.concatenate((NFGy, NHIy))

    N = np.concatenate((NFG, NHI))

    Nu = N.copy()
    Nv = N.copy()
    Nw = N.copy()
    Nbz = N.copy()
    Nbt = N.copy()

    nt = len(N)

    H00 = np.zeros(nt)

    # ------------------------------------------------------------------
    # Nuvw matrix
    # ------------------------------------------------------------------

    Nuvw = np.vstack([
        np.concatenate((Nu,  H00, H00, H00, H00)),
        np.concatenate((H00, Nv,  H00, H00, H00)),
        np.concatenate((H00, H00, Nw,  H00, H00)),
        np.concatenate((H00, H00, H00, Nbz, H00)),
        np.concatenate((H00, H00, H00, H00, Nbt))
    ])

    # ------------------------------------------------------------------
    # Penalty matrix
    # ------------------------------------------------------------------

    Kuvw_top = np.diag([
        Ku_top,
        Kv_top,
        Kw_top,
        Kbz_top,
        Kbt_top
    ])

    # ------------------------------------------------------------------
    # Boundary stiffness
    # ------------------------------------------------------------------

    I_Ktopa = Nuvw.T @ Kuvw_top @ Nuvw

    # The MATLAB code currently uses only this contribution.
    I_Ke = I_Ktopa

    return I_Ke