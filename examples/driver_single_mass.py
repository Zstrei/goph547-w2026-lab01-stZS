import numpy as np
import matplotlib.pyplot as plt

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


def compute_fields_on_grid(x_vals, y_vals, z_obs, xm, m):
    """
    Compute U and gz on an (x,y) grid at a fixed observation height z_obs.
    Returns U_grid, gz_grid with shape (ny, nx) matching meshgrid output.
    """
    X, Y = np.meshgrid(x_vals, y_vals)  # shape (ny, nx)
    U = np.zeros_like(X, dtype=float)
    gz = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):        # over y rows
        for j in range(X.shape[1]):    # over x cols
            x = np.array([X[i, j], Y[i, j], z_obs], dtype=float)
            U[i, j] = gravity_potential_point(x, xm, m)
            gz[i, j] = gravity_effect_point(x, xm, m)

    return X, Y, U, gz


def main():
    # Given in lab handout
    m = 1.0e7
    xm = np.array([0.0, 0.0, -10.0], dtype=float)
    z_list = [0.0, 10.0, 100.0]

    # Two grid spacings to compare
    dx_list = [5.0, 25.0]

    for dx in dx_list:
        x_vals = np.arange(-100.0, 100.0 + dx, dx)
        y_vals = np.arange(-100.0, 100.0 + dx, dx)

        # First compute all fields so we can set consistent colorbar limits
        results = []
        U_all = []
        gz_all = []

        for z_obs in z_list:
            X, Y, U, gz = compute_fields_on_grid(x_vals, y_vals, z_obs, xm, m)
            results.append((z_obs, X, Y, U, gz))
            U_all.append(U)
            gz_all.append(gz)

        U_all = np.array(U_all)
        gz_all = np.array(gz_all)

        Umin, Umax = float(U_all.min()), float(U_all.max())
        gzmin, gzmax = float(gz_all.min()), float(gz_all.max())

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), constrained_layout=True)

        for row, (z_obs, X, Y, U, gz) in enumerate(results):
            # U plot (left)
            axU = axes[row, 0]
            cU = axU.contourf(X, Y, U, levels=30, vmin=Umin, vmax=Umax, cmap="viridis")
            axU.plot(X, Y, "xk", markersize=2)
            axU.set_title(f"Gravity Potential U at z = {z_obs:.0f} m (dx = {dx:g} m)")
            axU.set_xlabel("x (m)")
            axU.set_ylabel("y (m)")
            fig.colorbar(cU, ax=axU)

            # gz plot (right)
            axG = axes[row, 1]
            cG = axG.contourf(X, Y, gz, levels=30, vmin=gzmin, vmax=gzmax, cmap="viridis")
            axG.plot(X, Y, "xk", markersize=2)
            axG.set_title(f"Gravity Effect gz at z = {z_obs:.0f} m (dx = {dx:g} m)")
            axG.set_xlabel("x (m)")
            axG.set_ylabel("y (m)")
            fig.colorbar(cG, ax=axG)

        fig.suptitle(f"Single Point Mass Anomaly (m={m:.1e} kg, xm={xm})", fontsize=14)
        plt.show()


if __name__ == "__main__":
    main()
