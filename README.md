# Blunomy Python Test

## Project Structure

```text
.
├── research/
│   ├── notebooks/            # Exploratory notebooks
│   └── src/research/         # Research helpers for loading, plotting, and clustering
├── solution/
│   ├── src/solution/         # Reusable solution package
│   └── tests/                # Pytest coverage for clustering and fitting
├── pixi.toml                 # Workspace dependencies, environments, and tasks
└── pixi.lock                 # Locked dependency versions
```

The `research` package contains the investigation and exploratory code. The `solution` package contains the reusable implementation: clustering wires, fitting catenary curves, and plotting fitted curves.

## Setup and Commands

This repository uses [Pixi](https://pixi.sh/) to manage Python environments and dependencies.

Install the project environments:

```bash
pixi install
```

Run Jupyter Lab for the research notebooks:

```bash
pixi run -e research lab
```

Run the solution test suite:

```bash
pixi run -e solution pytest
```

## Notebook Summary

```text
.
├── research/
│   └── notebooks/
│       ├── 01_data_inspection      # Inspect and illustrate the dataset
│       ├── 02_clustering           # Explore using HDBSCAN to cluster the points
│       ├── 03_curve_fit            # Use PCA to get the along-wire direction, find the plane for each wire, and use least squares to fit the curve
│       └── 04_outcome              # Demonstrate how to use Solution package, and illustrate a limitation of the method
```

## Usage

To use the `fit` function in the solution package, first prepare a `pd.DataFrame` containing the columns `x`, `y`, and `z`. Calling `fit` returns a list of `FitResult` objects with the fields shown below. The meaning of each field is explained in the `03_curve_fit` notebook.

```python
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
```

Here is a small usage example, similar to the one in the `04_outcome` notebook:

```python
from solution import fit, plot_fitted_curve
import matplotlib.pyplot as plt

# data is a pd.DataFrame with columns x, y, z
fit_results = fit(data)

ax = plt.figure().add_subplot(projection="3d")

# Draw the original points
ax.scatter(data["x"], data["y"], data["z"], c = "blue", alpha = 0.1, s = 2)

# Draw the fitted curves
for result in fit_results:
    plot_fitted_curve(ax, result)

ax.view_init(elev=50, azim=25)
plt.show()
```
