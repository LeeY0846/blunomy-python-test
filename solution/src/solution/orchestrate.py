from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from .cluster import cluster_wires
from .wire_fit import FitResult, catenary, fit_curve

def fit(data: pd.DataFrame, cluster_method = "hdbscan") -> list[FitResult]:
    """Cluster wire points and fit a catenary curve using least squares to each detected wire."""
    wires = cluster_wires(data, cluster_method)
    fit_results = []
    for wire in wires:
        fit_results.append(fit_curve(wire.to_numpy()))

    return fit_results

def plot_fitted_curve(ax: Axes3D, fit_result:FitResult):
    """Plot a fitted catenary curve on a 3D matplotlib axis."""
    u_line = np.linspace(fit_result.u_start, fit_result.u_end, 400)
    v_line = catenary(u_line, fit_result.c_fit, fit_result.u0_fit, fit_result.v0_fit)

    curve_3d = fit_result.origin + np.outer(u_line, fit_result.e1) + np.outer(v_line, fit_result.e2)

    ax.scatter(curve_3d[:,0], curve_3d[:,1], curve_3d[:,2], c = "red", s = 1)
