from second_disorder.base import HistogramGrid
import numpy as np

def run_add_histograms(grids, out):
    grids = [HistogramGrid.from_npz_name(f) for f in grids]
    out = grids[0]
    for other in grids[1:]:
        out.central += other.central
        out.hist += other.hist
        assert np.allclose(out.grid.origin, other.grid.origin)
        assert np.allclose(out.grid.delta, other.grid.delta)
        assert np.allclose(out.grid.shape, other.grid.shape)
        assert np.allclose(out.binning.number, other.binning.number)
        assert np.allclose(out.binning.spacing, other.binning.spacing)
    out.save_npz(out)
