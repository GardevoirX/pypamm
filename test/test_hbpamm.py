from numpy import allclose

from pysrc.hbpamm import HBPAMM

EXPECTED_COORD = [
    [2.30202112,  4.25360095, 2.69259933,], 
    [2.49967554,  4.45125537, 2.99197918,], 
    [0.62291489,  2.57449472, 2.56943865,], 
    [2.03362905,  3.98520888, 2.69176492,], 
    [-2.30202112, 4.25360095, 2.69259933,],
    [-2.49967554, 4.45125537, 2.99197918,],
    [-0.62291489, 2.57449472, 2.56943865,],
    [-2.03362905, 3.98520888, 2.69176492,],
    [0.57911912,  2.72515528, 2.69259933,], 
    [2.20858405,  4.35462021, 2.99197918,], 
]

EXPECTED_WEIGHTS = [0.02902873, 0.02463838, 0.06237020,
                    0.03162743, 0.02902873, 0.02463838,
                    0.06237020, 0.03162743, 0.05237393,
                    0.02372949]

EXPECTED_LEN = 272330

def test_basic_hbpamm():

    runner = HBPAMM(ta='O', td='O', th='H', cutoff=4.5, 
                    weighted=True, 
                    box='12.4244, 12.4244, 12.4244, 90, 90, 90')
    result = runner.run('examples/water/h2o-blyp-piglet.xyz')
    assert len(result[0]) == len(result[1])
    assert len(result[0]) == EXPECTED_LEN
    assert allclose(result[0][:10], EXPECTED_COORD)
    assert allclose(result[1][:10], EXPECTED_WEIGHTS)

