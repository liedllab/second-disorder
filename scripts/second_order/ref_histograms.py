#!/usr/bin/env python3

import gisttools as gt
import numpy as np
from base import RadialBinning, HistogramGrid

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dens2')
    parser.add_argument('--out', required=True)
    parser.add_argument('--hist_spacing', default=.0125, type=float)
    parser.add_argument('--hist_bins', default=80, type=float)
    args = parser.parse_args()
    binning = RadialBinning(args.hist_bins, args.hist_spacing)
    dens2 = load_density(args.dens2, 1, 1.)
    grid = dens2.grid
    all_gsvp = HistogramGrid.empty(grid, binning)
    flat_dens2 = dens2.data.values.ravel()
    binvol = binning.volume
    for i_voxel in range(grid.size):
        if i_voxel % 100 == 0:
            print(f'{i_voxel/grid.size:.03f}', end='\r')
        center = grid.xyz(i_voxel)[0]
        density = flat_dens2[i_voxel]
        Gsvp = expected_radial_pop_per_frame(
            grid=grid,
            center=center,
            dens=flat_dens2,
            binning=binning,
        )
        nans = np.isnan(Gsvp)
        Gsvp[nans] = density * binvol[nans]
        all_gsvp.semi_flat_hist()[i_voxel] = Gsvp
        all_gsvp.flat_central()[i_voxel] = density
    all_gsvp.save_npz(args.out)
    print('Done')
    return

        
def load_density(fname, n_frames, rho0):
    gistfile = gt.gist.load_dx(fname, colname='dens')
    gistfile['dens'] = gistfile['dens'] / n_frames / gistfile.grid.voxel_volume / rho0
    return gistfile


def entropy_contrib(a):
    with np.errstate(invalid='ignore', divide='ignore'):
        log_a = np.log(a)
    log_a[a == 0] = 0
    return a * log_a - a + 1


def radial_sums(grid, center, pop, binning):
    ind, dist = grid.surrounding_sphere(center, binning.limit)
    discrete = binning.assign(dist)
    counts = np.bincount(discrete)[:binning.number]
    total = np.bincount(discrete, weights=pop[ind])[:binning.number]
    return counts, total


def expected_radial_pop_per_frame(grid, center, dens, binning):
    averaged = radial_average(grid, center, dens, binning)
    return averaged * binning.volume


def radial_average(grid, center, pop, binning):
    counts, total = radial_sums(grid, center, pop, binning)
    return total / counts


def shifted_linspace(start, stop, n):
    delta = (stop-start)/n
    return np.linspace(start, stop, n, endpoint=False) + delta/2


def test_shifted_linspace():
    assert np.allclose(shifted_linspace(0, 1, 5), np.array([.1, .3, .5, .7, .9]))
    

if __name__ == '__main__':
    test_shifted_linspace()
    main()
