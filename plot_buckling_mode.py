
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)

def plot_cylinder_buckling_mode(qqw, range_x, range_y, L, Rc,
                                 n_z=250, n_theta=250,
                                 scale=None, cmap='jet'):
    """
    Computes the radial displacement field W(theta, z) from the modal DOF
    vector qqw (matching the NFG/NHI ordering used in evaluate_KF.py) and
    plots the buckled cylinder shape in 3D:
        (x, y, z) = ((Rc + scale*w)*cos(theta), (Rc + scale*w)*sin(theta), z)
    colored by the radial displacement w.

    scale : deformation amplification factor. If None, auto-scaled so the
            max deformation is ~10% of Rc (purely for visualization).
    """
    z = np.linspace(0, L, n_z)
    theta = np.linspace(0, 2 * np.pi, n_theta)

    # --- compute W(theta, z) ---
    rx = np.asarray(range_x).flatten()   # (nx,)
    ry = np.asarray(range_y).flatten()   # (ny,)
    nx_, ny_ = len(rx), len(ry)
    n_modes = nx_ * ny_

    F = np.sin(np.outer(rx, np.pi * z / L))   # (nx, nz)
    H = F
    G = np.sin(np.outer(ry, theta))           # (ny, ntheta)
    I = np.cos(np.outer(ry, theta))           # (ny, ntheta)

    qqw_FG = qqw[:n_modes].reshape(ny_, nx_)
    qqw_HI = qqw[n_modes:].reshape(ny_, nx_)

    W_FG = np.einsum('ki,lj,kl->ij', G, F, qqw_FG)
    W_HI = np.einsum('ki,lj,kl->ij', I, H, qqw_HI)

    W = W_FG + W_HI   # (n_theta, n_z)

    # normalize mode shape (eigenvectors are defined up to a scale factor)
    W = W / np.max(np.abs(W))

    if scale is None:
        scale = 0.1 * Rc   # visual amplification, NOT physical units

    # --- build 3D surface ---
    Z, Theta = np.meshgrid(z, theta)
    R = Rc + scale * W

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(W.min(), W.max())
    cmap_obj = plt.colormaps[cmap]
    surf = ax.plot_surface(X, Y, Z, facecolors=cmap_obj(norm(W)),
                            rstride=1, cstride=1, linewidth=0, antialiased=True,
                            shade=False)

    mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    mappable.set_array(W)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='Normalized w')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Cylinder buckling mode shape')
    ax.set_box_aspect([1, 1, L / (2 * Rc)])  # keep aspect ratio realistic

    plt.tight_layout()
    plt.show()

# --- usage in your existing script ---
#plot_cylinder_buckling_mode(qqw, range_x, range_y, L, Rc)