import second_disorder.base as sd_base
import numpy as np

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
