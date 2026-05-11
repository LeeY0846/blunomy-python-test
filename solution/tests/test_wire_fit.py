from tests.read_data import EXPECTED_RESULTS, get_test_data
from tests.assert_helpers import assert_fit_result_list_close
from solution.cluster import cluster_wires
from solution.wire_fit import fit_curve

def test_fit_results():
    test_data = get_test_data()

    wires = []
    actual_results = []

    for data in test_data:
        wires = cluster_wires(data, "hdbscan")
        for wire in wires:
            actual_results.append(fit_curve(wire.to_numpy()))
            
    assert_fit_result_list_close(actual_results, EXPECTED_RESULTS, atol=1e-3, rtol=1e-3)