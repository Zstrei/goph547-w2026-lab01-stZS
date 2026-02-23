import sys
from pathlib import Path

# Allow import of goph547lab01 package
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from goph547lab01.gravity import gravity_effect_point, gravity_potential_point


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_anomaly(mat_path: Path):
    data = loadmat(str(mat_path))
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    z = np.asarray(data["z"], dtype=float)
    rho = np.asarray(data["rho"], dtype=float)
    return x, y, z, rho


def integrate_mass_and_barycentre(x, y, z, rho, dx_cell=2.0):
    dV = dx_cell**3
    m_cells = rho * dV

    M = float(np.sum(m_cells))

    xb = float(np.sum(m_cells * x) / M)
    yb = float(np.sum(m_cells * y) / M)
    zb = float(np.sum(m_cells * z) / M)

    rho_max = float(np.max(rho))
    rho_mean = float(np.mean(rho))

    return M, np.array([xb, yb, zb]), rho_max, rho_mean


def mean_density_sections(x, y, z, rho):
    rho_yz = rho.mean(axis=0)
    rho_xz = rho.mean(axis=1)
    rho_xy = rho.mean(axis=2)

    Y_yz = y.mean(axis=0)
    Z_yz = z.mean(axis=0)

    X_xz = x.mean(axis=1)
    Z_xz = z.mean(axis=1)

    X_xy = x.mean(axis=2)
    Y_xy = y.mean(axis=2)

    return (X_xz, Z_xz, rho_xz), (Y_yz, Z_yz, rho_yz), (X_xy, Y_xy, rho_xy)


def pick_non_negligible_region(x, y, z, rho, frac=0.10):
    threshold = frac * float(np.max(rho))
    mask = rho >= threshold

    mean_region = float(np.mean(rho[mask]))

    xr = (float(x[mask].min()), float(x[mask].max()))
    yr = (float(y[mask].min()), float(y[mask].max()))
    zr = (float(z[mask].min()), float(z[mask].max()))

    return threshold, mean_region, xr, yr, zr


def forward_model_density(x, y, z, rho, x_s, y_s, z_obs, dx_cell=2.0):
    dV = dx_cell**3
    m_cells = (rho * dV).ravel()
    r_cells = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    X, Y = np.meshgrid(x_s, y_s)
    U = np.zeros_like(X)
    gz = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = np.array([X[i, j], Y[i, j], z_obs], dtype=float)
            U_total = 0.0
            gz_total = 0.0

            for rk, mk in zip(r_cells, m_cells):
                U_total += gravity_potential_point(p, rk, mk)
                gz_total += gravity_effect_point(p, rk, mk)

            U[i, j] = U_total
            gz[i, j] = gz_total

    return X, Y, U, gz


def second_derivative_x(F, dx):
    d2 = np.full_like(F, np.nan)
    d2[:, 1:-1] = (F[:, 2:] - 2*F[:, 1:-1] + F[:, :-2]) / dx**2
    return d2


def second_derivative_y(F, dy):
    d2 = np.full_like(F, np.nan)
    d2[1:-1, :] = (F[2:, :] - 2*F[1:-1, :] + F[:-2, :]) / dy**2
    return d2


def main():
    script_dir = Path(__file__).resolve().parent
    outputs = ensure_dir(script_dir.parent / "outputs")

    mat_path = script_dir / "anomaly_data.mat"
    if not mat_path.exists():
        raise FileNotFoundError("anomaly_data.mat must be inside examples/")

    x, y, z, rho = load_anomaly(mat_path)

    # ---- 1. Integrated properties ----
    M, rbar, rho_max, rho_mean = integrate_mass_and_barycentre(x, y, z, rho)

    print("\n=== Integrated Properties ===")
    print(f"Total mass: {M:.6e} kg")
    print(f"Barycentre (x,y,z): {rbar}")
    print(f"Max cell density: {rho_max:.6e} kg/m^3")
    print(f"Mean overall density: {rho_mean:.6e} kg/m^3")

    # ---- 2. Mean density cross-sections ----
    (X_xz, Z_xz, R_xz), (Y_yz, Z_yz, R_yz), (X_xy, Y_xy, R_xy) = mean_density_sections(x, y, z, rho)

    vmin = min(R_xz.min(), R_yz.min(), R_xy.min())
    vmax = max(R_xz.max(), R_yz.max(), R_xy.max())

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

    c0 = axes[0].contourf(X_xz, Z_xz, R_xz, 30, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].plot(rbar[0], rbar[2], "xk", markersize=3)
    axes[0].set_title("Mean Density (xz)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    fig.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(Y_yz, Z_yz, R_yz, 30, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].plot(rbar[1], rbar[2], "xk", markersize=3)
    axes[1].set_title("Mean Density (yz)")
    axes[1].set_xlabel("y (m)")
    axes[1].set_ylabel("z (m)")
    fig.colorbar(c1, ax=axes[1])

    c2 = axes[2].contourf(X_xy, Y_xy, R_xy, 30, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[2].plot(rbar[0], rbar[1], "xk", markersize=3)
    axes[2].set_title("Mean Density (xy)")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    fig.colorbar(c2, ax=axes[2])

    fig.savefig(outputs / "mean_density_sections.png", dpi=300)
    plt.close(fig)

    # ---- 3. Non-negligible region ----
    threshold, mean_region, xr, yr, zr = pick_non_negligible_region(x, y, z, rho)

    print("\n=== Non-negligible Region ===")
    print(f"Threshold used: {threshold:.6e}")
    print(f"x range: {xr}")
    print(f"y range: {yr}")
    print(f"z range: {zr}")
    print(f"Mean density in region: {mean_region:.6e}")

    # ---- 4 & 5. Forward modelling ----
    dx_s = 5.0
    x_s = np.arange(x.min()-10, x.max()+10, dx_s)
    y_s = np.arange(y.min()-10, y.max()+10, dx_s)

    z_levels = [0.0, 1.0, 100.0, 110.0]
    gz_maps = {}

    for zz in z_levels:
        X, Y, U, gz = forward_model_density(x, y, z, rho, x_s, y_s, zz)
        gz_maps[zz] = gz

    # ---- 6. Plot gz elevations ----
    gz_stack = np.stack([gz_maps[z] for z in z_levels])
    vmin_gz = gz_stack.min()
    vmax_gz = gz_stack.max()

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, zz in zip(axes, z_levels):
        c = ax.contourf(X, Y, gz_maps[zz], 30, vmin=vmin_gz, vmax=vmax_gz, cmap="viridis")
        ax.plot(X.ravel(), Y.ravel(), "xk", markersize=2)
        ax.set_title(f"gz at z={zz} m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(c, ax=ax)

    fig.savefig(outputs / "gz_4_elevations.png", dpi=300)
    plt.close(fig)

    # ---- 7. Second derivative via Laplace ----
    d2z0 = -(second_derivative_x(gz_maps[0.0], dx_s) +
             second_derivative_y(gz_maps[0.0], dx_s))

    d2z100 = -(second_derivative_x(gz_maps[100.0], dx_s) +
               second_derivative_y(gz_maps[100.0], dx_s))

    print("\nDone.")
    

if __name__ == "__main__":
    main()