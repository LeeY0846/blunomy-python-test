import numpy as np
import pandas as pd
import pytest
from tests.read_data import get_test_data
from solution.cluster import cluster_hdbscan, cluster_wires

def test_cluster_hdbscan():
    answers = [3,7,3,3]

    test_data = get_test_data()

    for i, df in enumerate(test_data):
        labels = cluster_hdbscan(df)
        assert np.unique(labels).size == answers[i]

def test_cluster_wires():
    answers = [3,7,3,3]
    
    test_data = get_test_data()

    for i, df in enumerate(test_data):
        wires = cluster_wires(df, "hdbscan")
        assert len(wires)== answers[i]

def test_invalid_cluster_wires_methods():
    df = pd.DataFrame()

    with pytest.raises(AttributeError):
        cluster_wires(df, "some_other_method")