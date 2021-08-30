import gisttools as gt
import numpy as np
import pytest

import second_disorder.base as sd_base

def test_binning_assign():
    binning = sd_base.RadialBinning(40, .025)
    assert binning.assign(0.001) == 0
    assert binning.assign(10000000) == binning.number

def test_rebin():
    arr = np.arange(16).reshape(4, 4)
    expected_1d = np.array([[4, 6, 8, 10], [20, 22, 24, 26]])
    expected_2d = np.array([[10, 18], [42, 50]])
    np.testing.assert_array_equal(sd_base.rebin(arr, 2, axes=(0,)), expected_1d)
    np.testing.assert_array_equal(sd_base.rebin(arr, 2, axes=(0, 1)), expected_2d)
    return

@pytest.fixture
def simple_histogram_grid():
    hist = np.zeros((2, 2, 2, 2))
    occurrences = np.zeros((2, 2, 2))
    # This would be strange since the hist is not zero where the occurrences are.
    hist[:, :, :, 0] = 3. 
    hist[0, 0, 0, 1] = 4
    hist[0, 1, 0, 1] = 2
    occurrences[0, 0, 0] = 1
    occurrences[0, 1, 0] = 2
    occurrences[1, 1, 0] = 1
    grid = gt.grid.Grid(origin=-.5, shape=(2, 2, 2), delta=1)
    binning = sd_base.RadialBinning(2, 10.)
    return sd_base.HistogramGrid(hist, occurrences, grid, binning)

def test_coarse_grain(simple_histogram_grid):
    coarse = simple_histogram_grid.coarse_grain(2)
    assert np.allclose(coarse.hist, [[[[24, 6]]]])

def test_coarse_grain_to_compute_mean(simple_histogram_grid):
    occ = simple_histogram_grid.occurrences.copy()
    simple_histogram_grid.occurrences[:] = 1
    simple_histogram_grid.rescale(occ)
    mean = simple_histogram_grid.coarse_grain(2)
    mean.rescale(1)
    assert mean.hist[0, 0, 0, 0] == 3
    assert mean.hist[0, 0, 0, 1] == 2
