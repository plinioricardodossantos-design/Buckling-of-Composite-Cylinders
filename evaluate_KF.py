import numpy as np

def evaluate_KF(x, theta, C, a, b, range_z, range_theta):
    """
    Python version of the MATLAB function evaluate_KF.

    Parameters
    ----------
    x : float
    theta : float
    C : ndarray (8x8)
    a : float
    b : float
    range_z : ndarray
    range_theta : ndarray

    Returns
    -------
    I_K : ndarray
    I_F : ndarray
    """

    Rc = b

    # Convert to numpy arrays
    range_z = np.asarray(range_z)
    range_theta = np.asarray(range_theta)

    # ------------------------------------------------------------------
    # FG
    # ------------------------------------------------------------------
    F = np.sin(range_z * np.pi * x / a)
    G = np.sin(range_theta * theta)

    #NFGx = (range_z * np.pi / a) * np.cos(range_z * np.pi * x / a) * np.sin(range_theta * theta)
    #NFGy = F * range_theta * np.cos(range_theta * theta)

    # ------------------------------------------------------------------
    # HI
    # ------------------------------------------------------------------
    H = np.sin(range_z * np.pi * x / a)
    I = np.cos(range_theta * theta)

    #NHIx = (range_z * np.pi / a) * np.cos(range_z * np.pi * x / a) * np.cos(range_theta * theta)
    #NHIy = -H * range_theta * np.sin(range_theta * theta)

    # ------------------------------------------------------------------
    # Shape functions
    # ------------------------------------------------------------------
    NFG = np.outer(G, F).reshape(-1)
    NHI = np.outer(I, H).reshape(-1)

    NFGx = np.outer(np.sin(range_theta * theta),
                    (range_z * np.pi / a) * np.cos(range_z * np.pi * x / a)).reshape(-1)

    NFGy = np.outer(range_theta * np.cos(range_theta * theta),
                    F).reshape(-1)

    NHIx = np.outer(np.cos(range_theta * theta),
                    (range_z * np.pi / a) * np.cos(range_z * np.pi * x / a)).reshape(-1)

    NHIy = np.outer(-range_theta * np.sin(range_theta * theta),
                    H).reshape(-1)

    # ------------------------------------------------------------------
    # Assemble arrays
    # ------------------------------------------------------------------
    Nx = np.concatenate((NFGx, NHIx))
    Ny = np.concatenate((NFGy, NHIy))
    N = np.concatenate((NFG, NHI))

    Nu = N.copy()
    Nv = N.copy()
    Nw = N.copy()

    Nux = Nx
    Nuy = Ny
    Nvx = Nx
    Nvy = Ny

    Nbzx = Nx
    Nbty = Ny
    Nbzy = Ny
    Nbtx = Nx

    Nwx = Nx
    Nwy = Ny

    Nbz = N
    Nbt = N

    nt = len(N)

    H00 = np.zeros(nt)

    H11 = Nux
    H22 = Nvy / Rc
    H23 = Nw / Rc
    H31 = Nuy / Rc
    H32 = Nvx

    H44 = Nbzx
    H55 = Nbty / Rc
    H64 = Nbzy / Rc
    H65 = Nbtx

    H73 = Nwx
    H74 = Nbz

    H82 = -Nv / Rc
    H83 = Nwy / Rc
    H85 = Nbt

    # ------------------------------------------------------------------
    # H matrix
    # ------------------------------------------------------------------
    Hmat = np.vstack([
        np.concatenate((H11, H00, H00, H00, H00)),
        np.concatenate((H00, H22, H23, H00, H00)),
        np.concatenate((H31, H32, H00, H00, H00)),
        np.concatenate((H00, H00, H00, H44, H00)),
        np.concatenate((H00, H00, H00, H00, H55)),
        np.concatenate((H00, H00, H00, H64, H65)),
        np.concatenate((H00, H00, H73, H74, H00)),
        np.concatenate((H00, H82, H83, H00, H85))
    ])

    # ------------------------------------------------------------------
    # Nuvw matrix
    # ------------------------------------------------------------------
    Nuvw = np.vstack([
        np.concatenate((Nu, H00, H00, H00, H00)),
        np.concatenate((H00, Nv, H00, H00, H00)),
        np.concatenate((H00, H00, Nw, H00, H00)),
        np.concatenate((H00, H00, H00, Nbz, H00)),
        np.concatenate((H00, H00, H00, H00, Nbt))
    ])

    f = np.array([[0],
                  [0],
                  [0.001],
                  [0],
                  [0]])

    I_K = Hmat.T @ C @ Hmat
    I_F = Nuvw.T @ f

    return I_K, I_F