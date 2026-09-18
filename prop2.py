import numpy as np


def prop(cylinder):

    if cylinder == 1:  # User Defined

        L = 510
        Rc = 250
        hp = 0.125

        E1 = 123550
        E2 = 8708
        E3 = E2

        v12 = 0.319

        G12 = 5695
        G13 = G12
        G23 = 3400

        theta_deg = np.array([45, -45, 15, -15])

    elif cylinder == 2:  # Z11

        L = 510
        Rc = 250
        hp = 0.125

        E1 = 123550
        E2 = 8708
        E3 = E2

        v12 = 0.319

        G12 = 5695
        G13 = G12
        G23 = 3400

        theta_deg = np.array(
            [60, -60, 0, 0, 68, -68, 52, -52, 37, -37]
        )

    elif cylinder == 3:  # Z33

        L = 510
        Rc = 250
        hp = 0.125

        E1 = 123550
        E2 = 8708
        E3 = E2

        v12 = 0.319

        G12 = 5695
        G13 = G12
        G23 = G12

        theta_deg = np.array(
            [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
        )

    elif cylinder == 4:  # ITA**

        L = 140
        Rc = 235 / 2
        hp = 0.3160

        E1 = 51830
        E2 = 51830
        E3 = E2

        v12 = 0.064

        G12 = 2820
        G13 = G12
        G23 = G12

        theta_deg = np.array([0, 0])

    elif cylinder == 5:  # Shadmehri

        L = 10
        Rc = 100
        hp = 0.5

        E1 = 210290.09744
        E2 = E1 / 40
        E3 = E2

        v12 = 0.25

        G12 = 0.6 * E2
        G13 = G12
        G23 = G12

        theta_deg = np.array([10, -10])

    elif cylinder == 6:  # Tec

        L = 525
        Rc = 350 / 2
        hp = 0.37

        E1 = 30277
        E2 = 16308
        E3 = E2

        v12 = 0.42

        G12 = 4700
        G13 = G12
        G23 = G12

        theta_deg = np.array([15, -15, 15, -15, 15, -15])

    elif cylinder == 7:  # R16

        L = 500
        Rc = 250
        hp = 0.523 / 4

        E1 = 150200
        E2 = 8900
        E3 = E2

        v12 = 0.32

        G12 = 5100
        G13 = G12
        G23 = G12

        theta_deg = np.array([24, -24, 41, -41])

    elif cylinder == 8:  # ITA01 degraded

        L = 140
        Rc = 235 / 2
        hp = 0.3160

        E1 = 21796
        E2 = 21796
        E3 = 5108

        v12 = 0.099

        G12 = 1784
        G13 = G12
        G23 = G12

        theta_deg = np.array([0, 0])

    elif cylinder == 9:  # Paper for external pressure

        L = 20
        Rc = 10
        hp = 0.01/2

        E1 = 30*10**6
        E2 = 0.75*10**6
        E3 = E2

        v12 = 0.25

        G12 = 3.75*10**6
        G13 = G12
        G23 = G12

        theta_deg = np.array([0, 90])

    else:
        raise ValueError("Invalid cylinder option")

    v21 = v12 * E2 / E1

    return (
        E1, E2, v12, v21,
        G12, G13, G23,
        theta_deg, hp, L, Rc
    )