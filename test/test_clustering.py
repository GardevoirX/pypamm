import numpy as np
from pysrc.clustering import MinMaxClustering

def test_basic():

    clustering =  MinMaxClustering(3, [0])
    clustering.fit(np.arange(10, dtype=float).reshape(-1, 1))
    assert np.all(clustering.selected_idx_ == [0, 9, 4])

def test_period():

    clustering =  MinMaxClustering(3, [0], period=np.array([10.]))
    clustering.fit(np.arange(10, dtype=float).reshape(-1, 1))
    assert np.all(clustering.selected_idx_ == [0, 5, 2])
