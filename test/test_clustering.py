import numpy as np
from pysrc.clustering import MinMaxClustering

def test_basic():

    clustering =  MinMaxClustering(3, [0])
    clustering.fit(np.arange(10).reshape(-1, 1))
    assert np.all(clustering.selected_idx_ == [0, 9, 4])
