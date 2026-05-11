# Blunomy Python Test

This project explores and solves a wire fitting problem from LiDAR point-cloud data. The input datasets are Parquet files containing `x`, `y`, and `z` coordinates for points sampled from overhead wires.

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
