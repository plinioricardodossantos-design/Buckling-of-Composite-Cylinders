import numpy as np
import matplotlib.pyplot as plt

def plot_cylinder_buckling_contour(qqw, range_x, range_y, L, Rc=None,
                                    n_z=100, n_theta=100, cmap='jet'):
    """
    Computes the radial displacement field W(theta, z) point-by-point using
    the modal DOF vector qqw, and plots it as a 2D contour (unrolled cylinder
    surface: s = theta*Rc vs z).

    Rc : cylinder radius. If provided, the horizontal axis is the arc length
         s = theta * Rc (in the same units as L). If None, the horizontal
         axis is theta itself (radians).
    """
    # Grid
    z = np.linspace(0, L, n_z)
    theta = np.linspace(0, 2 * np.pi, n_theta)

    Z, Theta = np.meshgrid(z, theta)
    W = np.zeros_like(Z)

    for i in range(len(theta)):
        for j in range(len(z)):

            # Shape functions at one point
            Fu = np.sin(range_x * np.pi * z[j] / L)
            Gu = np.sin(range_y * theta[i])

            Hu = np.sin(range_x * np.pi * z[j] / L)
            Iu = np.cos(range_y * theta[i])

            NFG = (Fu @ Gu).T.reshape(1, -1)
            NHI = (Hu @ Iu).T.reshape(1, -1)

            Nw = np.hstack((NFG, NHI))

            W[i, j] = Nw @ qqw

    if Rc is not None:
        Y = Rc * Theta
    else:
        Y = Theta

    plt.figure(figsize=(9, 5))
    plt.contourf(Y, Z, W, levels=50, cmap=cmap)
    plt.colorbar(label='w')

    plt.xlabel('s = θR' if Rc is not None else r'$\theta$')
    plt.ylabel('z [mm]')

    plt.tight_layout()
    plt.show()

    return W


# --- usage ---
#W = plot_cylinder_buckling_contour(qqw, range_x, range_y, L, Rc)