from numpy import allclose, loadtxt

from pysrc.pamm import PAMM

DISCRIPTORFILE = 'tutorials/betahairpin/GLOBAL/colvar.wt.30cv.4'
GRIDFILE = 'tutorials/betahairpin/GLOBAL/ref.idxs'
OUTPUTFILE = 'out.txt'
FOUTPUTFILE = 'test/betahairpin/fp0.1-qs1.grid'

def test_basic_pamm():

    descriptors = loadtxt(DISCRIPTORFILE)
    runner = PAMM(descriptors,
                  dimension=30,
                  period_text='6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28,6.28',
                  ngrid=1000,
                  fpoints=0.1,
                  qs=1,
                  outputfile='fp0.1-qs1',
                  verbose=True)
    grid = loadtxt('examples/betahairpin/GLOBAL/ref.idxs', dtype=int)[:1000] - 1
    runner.fit(grid)
    runner.get_output(OUTPUTFILE)
    out = loadtxt(OUTPUTFILE)
    fout = loadtxt(FOUTPUTFILE)
    out[:, -9] += 1
    assert allclose(out[:, -5:], fout[:, -5:], atol=1e-3)
    assert allclose(out[:, -8:-7], fout[:, -8:-7], atol=1e-3)
    assert allclose(out[:, :-9], fout[:, :-9], atol=1e-3)

    gaussian_weights, gaussian_means, gaussian_cov = runner.generate_probability_model()
    assert gaussian_weights[0] == 0.000696735345839212
    assert gaussian_means[0][0] == -0.825863
    assert gaussian_cov[0][0][0] == 0.03026473072516357
