#!/usr/bin/env python
# -*- coding: utf-8 -*-

from base import HistogramGrid
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('grids', nargs='+')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    grids = [HistogramGrid.from_npz_name(f) for f in args.grids]
    out = grids[0]
    for other in grids[1:]:
        out.central += other.central
        out.hist += other.hist
        assert np.allclose(out.grid.origin, other.grid.origin)
        assert np.allclose(out.grid.delta, other.grid.delta)
        assert np.allclose(out.grid.shape, other.grid.shape)
        assert np.allclose(out.binning.number, other.binning.number)
        assert np.allclose(out.binning.spacing, other.binning.spacing)
    out.save_npz(args.out)

if __name__ == "__main__":
    main()
