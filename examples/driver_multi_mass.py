# examples/driver_multi_mass.py
import sys
from pathlib import Path

# Add project root to Python path so goph547lab01 can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


def generate_mass_set(rng, m_total, rcm_target, max_tries=20000):
    """
    Generate 5 masses/locations such that:
      sum(m_i) = m_total
      sum(m_i * r_i) / m_total = rcm_target

    Sampling rules (lab):
      masses ~ N(mu=m_total/5, sigma=m_total/100)
      x,y ~ N(mu=0, sigma=20)
      z ~ N(mu=-10, sigma=2)

    Method:
      sample first 4 masses/locations
      compute 5th mass and 5th location to enforce constraints
    Enforce:
      all masses > 0
      all z <= -1
    """
    mu_m = m_total / 5.0
    sig_m = m_total / 100.0

    for _ in range(max_tries):
        # First 4 masses
        m4 = rng.normal(loc=mu_m, scale=sig_m, size=4)
        if np.any(m4 <= 0.0):
            continue

        # First 4 locations
        x4 = rng.normal(loc=0.0, scale=20.0, size=4)
        y4 = rng.normal(loc=0.0, scale=20.0, size=4)
        z4 = rng.normal(loc=-10.0, scale=2.0, size=4)

        # Enforce z <= -1 for first 4
        if np.any(z4 > -1.0):
            continue

        # Compute 5th mass
        m5 = m_total - float(np.sum(m4))
        if m5 <= 0.0:
            continue

        # Compute 5th location from centroid constraint:
        # m_total*rcm_target = sum(m_i * r_i) = sum_4(m4*r4) + m5*r5
        sum_mr4 = np.array([
            float(np.sum(m4 * x4)),
            float(np.sum(m4 * y4)),
            float(np.sum(m4 * z4)),
        ])

        r5 = (m_total * rcm_target - sum_mr4) / m5

        # Enforce z5 <= -1
        if r5[2] > -1.0:
            continue

        masses = np.concatenate([m4, [m5]])
        locs = np.column_stack([
            np.concatenate([x4, [r5[0]]]),
            np.concatenate([y4, [r5[1]]]),
            np.concatenate([z4, [r5[2]]]),
        ])

        # Final verification
        m_check = float(np.sum(masses))
        rcm_check = (masses[:, None] * locs).sum(axis=0) / m_check

        if not np.isclose(m_check, m_total, atol=1e-6):
            continue
        if not np.allclose(rcm_check, rcm_target, atol=1e-6):
            continue
        if np.any(locs[:, 2] > -1.0):
            continue

        return masses, locs

    raise RuntimeError("Could not generate a valid mass set. Increase max_tries or check constraints.")


def save_mass_set_mat(filepath, masses, locs):
    savemat(str(filepath), {
        "m": masses,
        "xm": locs,
    })

def load_mass_set_mat(filepath):
    data = loadmat(str(filepath))
    masses = np.array(data["m"]).flatten()
    locs = np.array(data["xm"])
    return masses, locs


def compute_fields_on_grid_multi(x_vals, y_vals, z_obs, masses, locs):
    """
    Compute total U and gz on an (x,y) grid at fixed z_obs
    by linear superposition over multiple point masses.
    """
    X, Y = np.meshgrid(x_vals, y_vals)  # shape (ny, nx)
    U = np.zeros_like(X, dtype=float)
    gz = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):        # over y rows
        for j in range(X.shape[1]):    # over x cols
            x = np.array([X[i, j], Y[i, j], z_obs], dtype=float)

            U_ij = 0.0
            gz_ij = 0.0
            for mk, rk in zip(masses, locs):
                U_ij += gravity_potential_point(x, rk, mk)
                gz_ij += gravity_effect_point(x, rk, mk)

            U[i, j] = U_ij
            gz[i, j] = gz_ij

    return X, Y, U, gz


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    # ===== Matching Part A parameters =====
    m_total = 1.0e7
    rcm_target = np.array([0.0, 0.0, -10.0], dtype=float)

    z_list = [0.0, 10.0, 100.0]
    dx_list = [5.0, 25.0]

    # Same spatial extent as Part A script
    extent = 100.0

    rng = np.random.default_rng(547)

    script_dir = Path(__file__).resolve().parent
    outputs_dir = ensure_dir(script_dir.parent / "outputs")  # repo_root/outputs

    # ============================================================
    # Generate and save 3 sets of 5 masses
    # ============================================================
    for i in range(1, 4):
        masses, locs = generate_mass_set(rng, m_total, rcm_target)

        m_check = float(np.sum(masses))
        rcm_check = (masses[:, None] * locs).sum(axis=0) / m_check

        print(f"\nMass set {i}")
        print(f"  Total mass: {m_check:.6e} kg (target {m_total:.6e})")
        print(f"  Centre of mass: {rcm_check} (target {rcm_target})")
        print(f"  z range: min={locs[:,2].min():.3f}, max={locs[:,2].max():.3f} (must be <= -1)")
        print(f"  masses: {masses}")
        print(f"  locs (x,y,z):\n{locs}")

        mat_path = script_dir / f"mass_set_{i}.mat"
        save_mass_set_mat(mat_path, masses, locs)
        print(f"  Saved: {mat_path}")

    # ============================================================
    # For each set, plot U and gz similar to Part A
    # for dx = 5 and 25, and z = 0,10,100
    # ============================================================
    for dx in dx_list:
        x_vals = np.arange(-extent, extent + dx, dx)
        y_vals = np.arange(-extent, extent + dx, dx)

        for i in range(1, 4):
            masses, locs = load_mass_set_mat(script_dir / f"mass_set_{i}.mat")

            # Compute all z first to get consistent color limits within this (set, dx)
            results = []
            U_all = []
            gz_all = []

            for z_obs in z_list:
                X, Y, U, gz = compute_fields_on_grid_multi(x_vals, y_vals, z_obs, masses, locs)
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
                axU.set_title(f"U (set {i}) at z = {z_obs:.0f} m (dx = {dx:g} m)")
                axU.set_xlabel("x (m)")
                axU.set_ylabel("y (m)")
                fig.colorbar(cU, ax=axU)

                # gz plot (right)
                axG = axes[row, 1]
                cG = axG.contourf(X, Y, gz, levels=30, vmin=gzmin, vmax=gzmax, cmap="viridis")
                axG.plot(X, Y, "xk", markersize=2)
                axG.set_title(f"gz (set {i}) at z = {z_obs:.0f} m (dx = {dx:g} m)")
                axG.set_xlabel("x (m)")
                axG.set_ylabel("y (m)")
                fig.colorbar(cG, ax=axG)

            fig.suptitle(f"Multiple Point Masses (set {i}) — total m={m_total:.1e} kg, COM={rcm_target}", fontsize=14)

            dx_tag = str(dx).replace(".", "p")
            outname = outputs_dir / f"multi_mass_set{i}_dx{dx_tag}.png"
            fig.savefig(outname, dpi=300)
            plt.close(fig)

            print(f"Saved plot: {outname}")


if __name__ == "__main__":
    main()
