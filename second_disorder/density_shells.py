#!/usr/bin/env python3

import gisttools as gt
import numpy as np
from second_disorder.base import RadialBinning, HistogramGrid

def run_density_shells(dens2, out, hist_spacing, hist_bins):
    binning = RadialBinning(hist_bins, hist_spacing)
    dens2 = load_density(dens2, 1, 1.)
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
        # all_gsvp.flat_central()[i_voxel] = density
        all_gsvp.flat_central()[i_voxel] = 1
    all_gsvp.save_npz(out)
    print('Done')
    return


def load_density(fname, n_frames, rho0):
    gistfile = gt.gist.load_dx(fname, colname='dens')
    gistfile['dens'] = gistfile['dens'] / n_frames / gistfile.grid.voxel_volume / rho0
    return gistfile


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


if __name__ == '__main__':
    main()
