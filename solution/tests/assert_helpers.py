from pathlib import Path
import pandas as pd

DIFFICULTIES = ["easy","medium","hard","extrahard"]

data_dir = Path(__file__).resolve().parents[0] / "data"

def read_dataset(difficulty: str):
    return read_parquet(data_dir / f"lidar_cable_points_{difficulty}.parquet")

def read_parquet(filename: str):
    return pd.read_parquet(filename, "auto")

def get_test_data():
    test_data = []

    for difficulty in DIFFICULTIES:
        test_data.append(read_dataset(difficulty))

    return test_data

import numpy as np


def assert_close_scalar(actual, expected, atol=1e-3, rtol=1e-3, name="value"):
    assert np.isclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name} mismatch: actual={actual}, expected={expected}, "
        f"atol={atol}, rtol={rtol}"
    )


def assert_close_array(actual, expected, atol=1e-3, rtol=1e-3, name="array"):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    assert actual.shape == expected.shape, (
        f"{name} shape mismatch: actual shape={actual.shape}, "
        f"expected shape={expected.shape}"
    )

    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name} mismatch:\n"
        f"actual   = {actual}\n"
        f"expected = {expected}\n"
        f"abs diff = {np.abs(actual - expected)}\n"
        f"atol={atol}, rtol={rtol}"
    )


def assert_close_direction(actual, expected, atol=1e-3, rtol=1e-3, name="direction"):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    assert actual.shape == expected.shape, (
        f"{name} shape mismatch: actual shape={actual.shape}, "
        f"expected shape={expected.shape}"
    )

    same = np.allclose(actual, expected, atol=atol, rtol=rtol)
    flipped = np.allclose(actual, -expected, atol=atol, rtol=rtol)

    assert same or flipped, (
        f"{name} mismatch up to sign:\n"
        f"actual    = {actual}\n"
        f"expected  = {expected}\n"
        f"-expected = {-expected}\n"
        f"atol={atol}, rtol={rtol}"
    )


def assert_fit_result_close(actual, expected, atol=1e-3, rtol=1e-3):
    assert_close_direction(actual.e1, expected.e1, atol=atol, rtol=rtol, name="e1")
    assert_close_direction(actual.e2, expected.e2, atol=atol, rtol=rtol, name="e2")
    assert_close_array(actual.origin, expected.origin, atol=atol, rtol=rtol, name="origin")

    assert_close_scalar(actual.c_fit, expected.c_fit, atol=atol, rtol=rtol, name="c_fit")
    assert_close_scalar(actual.u0_fit, expected.u0_fit, atol=atol, rtol=rtol, name="u0_fit")
    assert_close_scalar(actual.v0_fit, expected.v0_fit, atol=atol, rtol=rtol, name="v0_fit")
    assert_close_scalar(actual.u_start, expected.u_start, atol=atol, rtol=rtol, name="u_start")
    assert_close_scalar(actual.u_end, expected.u_end, atol=atol, rtol=rtol, name="u_end")


def assert_fit_result_list_close(actual_list, expected_list, atol=1e-3, rtol=1e-3):
    assert len(actual_list) == len(expected_list), (
        f"Length mismatch: actual={len(actual_list)}, expected={len(expected_list)}"
    )

    for i, (actual, expected) in enumerate(zip(actual_list, expected_list)):
        try:
            assert_fit_result_close(actual, expected, atol=atol, rtol=rtol)
        except AssertionError as e:
            raise AssertionError(f"FitResult at index {i} does not match:\n{e}") from e