import numpy as np

def evaluate_KGz(x, theta, a, b, range_z, range_theta):

    F = np.sin(range_z * np.pi * x / a)
    #G = np.sin(range_theta * theta)
    H = np.sin(range_z * np.pi * x / a)
    #I = np.cos(range_theta * theta)

    NFGx = (range_z * np.pi *np.cos(np.pi * range_z * x / a) * np.sin(range_theta * theta)) / a

    NHIx = (
        np.cos(np.pi * range_z * x / a) *
        range_z * np.pi *
        np.cos(range_theta * theta)
    ) / a

    NFGt = np.outer(range_theta * np.cos(range_theta * theta),
                    F).reshape(-1)

    NHIt = np.outer(-range_theta * np.sin(range_theta * theta),
                    H).reshape(-1)
    
    NFGx = NFGx.T.reshape(-1)
    NHIx = NHIx.T.reshape(-1)

    NFGt = NFGt.T.reshape(-1)
    NHIt = NHIt.T.reshape(-1)

    Nx = np.concatenate([NFGx, NHIx])
    Nt = np.concatenate([NFGt, NHIt])

    nt = len(Nx)

    H00 = np.zeros(nt)

    HGz = np.block([
        [H00, H00, H00, H00, H00],
        [H00, H00, H00, H00, H00],
        [H00, H00, Nx,  H00, H00],
        [H00, H00, H00, H00, H00],
        [H00, H00, H00, H00, H00] ])

    HGt = np.block([
        [H00, H00, H00, H00, H00],
        [H00, H00, H00, H00, H00],
        [H00, H00, Nt,  H00, H00],
        [H00, H00, H00, H00, H00],
        [H00, H00, H00, H00, H00] ])

    HG_Total = HGz.T @ HGz + 2*HGt.T @ HGt

    f_theta = 1.0

    #return f_theta * HGz.T @ HGz
    return f_theta * HG_Total.T @ HG_Total