import numpy as np


def abd(E1, E2, v12, G12, G13, G23, Np, theta_deg, hp):

    theta = np.radians(theta_deg)

    v21 = v12 * E2 / E1

    Q11 = E1 / (1 - v21 * v12)
    Q22 = E2 / (1 - v21 * v12)
    Q12 = v12 * E2 / (1 - v21 * v12)
    Q66 = G12

    Q = np.array([
        [Q11, Q12, 0],
        [Q12, Q22, 0],
        [0,   0,   Q66]
    ])

    Qs = np.array([
        [G13, 0],
        [0, G23]
    ])

    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    As = np.zeros((2, 2))

    Ks = 5 / 6

    for k in range(Np):

        ang = theta[k]

        T = np.array([
            [np.cos(ang)**2,
             np.sin(ang)**2,
             2*np.sin(ang)*np.cos(ang)],

            [np.sin(ang)**2,
             np.cos(ang)**2,
             -2*np.sin(ang)*np.cos(ang)],

            [-np.sin(ang)*np.cos(ang),
             np.sin(ang)*np.cos(ang),
             np.cos(ang)**2 - np.sin(ang)**2]
        ])

        Ts = np.array([
            [np.cos(ang), np.sin(ang)],
            [-np.sin(ang), np.cos(ang)]
        ])

        Q_off = np.linalg.inv(T) @ Q @ np.linalg.inv(T).T
        Qs_off = np.linalg.inv(Ts) @ Qs @ np.linalg.inv(Ts).T

        z0 = (k - Np/2) * hp
        z1 = (k + 1 - Np/2) * hp

        A += (z1 - z0) * Q_off
        B += 0.5 * (z1**2 - z0**2) * Q_off
        D += (1/3) * (z1**3 - z0**3) * Q_off

        As += Ks * (z1 - z0) * Qs_off

    return A, B, D, As