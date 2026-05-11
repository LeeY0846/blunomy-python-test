import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import least_squares

class FitResult:
    def __init__(self, e1, e2, origin, c_fit, u0_fit, v0_fit, u_start, u_end):
        self.e1 = e1
        self.e2 = e2
        self.origin = origin
        self.c_fit = c_fit
        self.u0_fit = u0_fit
        self.v0_fit = v0_fit
        self.u_start = u_start
        self.u_end = u_end

def catenary(u, c, u0, v0):
    return v0 + c * (np.cosh((u - u0) / c) - 1.0)

def residuals(params, u, v):
    c, u0, v0 = params
    pred = v0 + c * (np.cosh((u-u0)/c) - 1.0)
    return pred - v

def fit_curve(points):
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

    return FitResult(e1, e2, origin, c_fit, u0_fit, v0_fit, u.min(), u.max())