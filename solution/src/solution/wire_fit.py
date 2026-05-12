from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from scipy.optimize import least_squares

@dataclass
class FitResult:
    e1: NDArray[np.float64]
    e2: NDArray[np.float64]
    origin: NDArray[np.float64]
    c_fit: float
    u0_fit: float
    v0_fit: float
    u_start: float
    u_end: float
    rmse_3d: float
    r2: float

def catenary(u: NDArray[np.float64], c: float, u0: float, v0: float):
    """Evaluate a catenary curve for the given local coordinates."""
    return v0 + c * (np.cosh((u - u0) / c) - 1.0)

def residuals(params: NDArray[np.float64], u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return catenary prediction errors for least-squares fitting."""
    c, u0, v0 = params
    pred = v0 + c * (np.cosh((u-u0)/c) - 1.0)
    return pred - v

def fit_curve(points: NDArray[np.float64]):
    """Fit a catenary curve to 3D points representing a single wire."""
    pca = PCA(n_components=3)
    pca.fit(points)

    # Get the along-wire direction
    e1 = pca.components_[0]
    e1 = e1 / np.linalg.norm(e1)

    # Get the local vertical direction
    z_world = np.array([0.0, 0.0, 1.0])
    e2 = z_world - np.dot(z_world, e1) * e1
    e2 = e2 / np.linalg.norm(e2)

    # Get the plan normal
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)

    # Project points onto the plane
    origin = points.mean(axis = 0)

    points_relative = points - origin
    dist_normal = points_relative @ n

    points_projected = points - np.outer(dist_normal, n)

    rel_proj = points_projected - origin
    u = rel_proj @ e1
    v = rel_proj @ e2

    # Get the initial guess (the lowest point)
    c0 = max((u.max() - u.min()) / 4, 1e-3)
    u0_0 = u[np.argmin(v)]
    v0_0 = v.min()
    x0 = np.array([c0, u0_0, v0_0])

    # Perform least squares
    result = least_squares(
        residuals,
        x0=x0,
        args=(u, v),
        bounds=([1e-6, u.min(), v.min() - 10],
                [1e6, u.max(), v.max() + 10]),
        loss="soft_l1"
    )

    c_fit, u0_fit, v0_fit = result.x

    # Evaluate the fit
    v_pred = catenary(u, c_fit, u0_fit, v0_fit)
    res_curve = v - v_pred

    fitted_points = origin + np.outer(u, e1) + np.outer(v_pred, e2)
    dist_3d = np.linalg.norm(points - fitted_points, axis=1)
    # Get RMSE in 3D space
    rmse_3d = float(np.sqrt(np.mean(dist_3d**2)))

    ss_res = np.sum(res_curve**2)
    ss_tot = np.sum((v - np.mean(v))**2)
    # Get R2 for the fitting
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return FitResult(e1, e2, origin, c_fit, u0_fit, v0_fit, u.min(), u.max(), rmse_3d, r2)
