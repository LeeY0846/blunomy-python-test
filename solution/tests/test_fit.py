import numpy as np
from tests.read_data import EXPECTED_RESULTS, get_test_data
from tests.assert_helpers import assert_fit_result_list_close
from solution.cluster import cluster_wires
from solution.wire_fit import FitResult, fit_curve
from solution import fit

def test_fit_results():
    test_data = get_test_data()

    actual_results = []

    for data in test_data:
        actual_results.extend(fit(data, "hdbscan"))
            
    assert_fit_result_list_close(actual_results, EXPECTED_RESULTS, atol=1e-3, rtol=1e-3)