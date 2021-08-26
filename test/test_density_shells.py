import second_disorder.density_shells as sd_densshell
import numpy as np

def test_shifted_linspace():
    assert np.allclose(sd_densshell.shifted_linspace(0, 1, 5), np.array([.1, .3, .5, .7, .9]))
