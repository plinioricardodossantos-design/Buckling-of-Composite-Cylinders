import numpy as np
import matplotlib.pyplot as plt


def plot_disp(Nu, Nv, Nw, L, Rc):
    """
    Post-processing for buckling displacement.

    Parameters
    ----------
    Nu, Nv, Nw : ndarray
        Shape function matrices evaluated on the plotting grid.
    L : float
        Cylinder length.
    Rc : float
        Cylinder radius.
    """

    Fat = 5.0

    # Load eigenvector
    qq = np.loadtxt("qq.txt")

    Lu = Nu.shape[1]
    Lv = Nv.shape[1]
    Lw = Nw.shape[1]

    qqu = qq[:Lu]
    qqv = qq[Lu:Lu + Lv]
    qqw = qq[Lu + Lv:Lu + Lv + Lw]

    # Displacements
    u = Nu @ qqu
    v = Nv @ qqv
    w = Nw @ qqw

    # -----------------------------------------
    # Surface plot
    # -----------------------------------------

    n_theta = 80
    n_z = 80

    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(0, L, n_z)

    TH, Z = np.meshgrid(theta, z)

    # Evaluate w on the plotting grid
    #
    # IMPORTANT:
    # Replace this line with the correct evaluation
    # of the Fourier expansion.
    #
    w_eval = w.reshape(n_z, n_theta)

    R = Rc + Fat * w_eval

    X = R * np.cos(TH)
    Y = R * np.sin(TH)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=plt.cm.jet(
            (w_eval - w_eval.min()) /
            (w_eval.max() - w_eval.min())
        ),
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D displacement - w")

    mappable = plt.cm.ScalarMappable(cmap="jet")
    mappable.set_array(w_eval)
    plt.colorbar(mappable)

    plt.savefig("Buckling_mode.png", dpi=300)

    # -----------------------------------------
    # u(z)
    # -----------------------------------------

    plt.figure()

    plt.plot(z, u[:n_z])

    plt.xlabel("z (mm)")
    plt.ylabel("u (mm)")
    plt.title("u")

    # -----------------------------------------
    # v(z)
    # -----------------------------------------

    plt.figure()

    plt.plot(z, v[:n_z])

    plt.xlabel("z (mm)")
    plt.ylabel("v (mm)")
    plt.title("v")

    # -----------------------------------------
    # w(z)
    # -----------------------------------------

    plt.figure()

    plt.plot(z, w_eval[:, n_theta // 2])

    plt.xlabel("z (mm)")
    plt.ylabel("w (mm)")
    plt.title("w")

    plt.show()