import pandas as pd
from .cluster import cluster_wires
from .wire_fit import FitResult, fit_curve

def fit(data: pd.DataFrame, cluster_method = "hdbscan") -> list[FitResult]:
    wires = cluster_wires(data, cluster_method)
    fit_results = []
    for wire in wires:
        fit_results.append(fit_curve(wire.to_numpy()))

    return fit_results