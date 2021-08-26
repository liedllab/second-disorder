import second_disorder.entropy as sd_ent
import numpy as np

def test_shifted_linspace():
    assert np.allclose(sd_ent.shifted_linspace(0, 1, 5), np.array([.1, .3, .5, .7, .9]))
