#!/usr/bin/env python3

import numpy as np
from second_disorder.base import RadialBinning, HistogramGrid, load_density, fixed_length_bincount

def run_density_shells(dens2, out, hist_spacing, hist_bins):
    binning = RadialBinning(hist_bins, hist_spacing)
    dens2 = load_density(dens2, 1.)
    grid = dens2.grid
    all_gsvp = HistogramGrid.empty(grid, binning)
    flat_dens2 = dens2.data.values.ravel()
    binvol = binning.volume
    for i_voxel in range(grid.size):
        if i_voxel % 100 == 0:
            print(f'{i_voxel/grid.size:.03f}', end='\r')
        center = grid.xyz(i_voxel)[0]
        density = flat_dens2[i_voxel]
        Gsvp = radial_average(
            grid=grid,
            center=center,
            dens=flat_dens2,
            binning=binning,
        )
        nans = np.isnan(Gsvp)
        Gsvp[nans] = density * binvol[nans]
        all_gsvp.semi_flat_hist()[i_voxel] = Gsvp
        all_gsvp.flat_occurrences()[i_voxel] = 1
    all_gsvp.save_npz(out)
    print('Done')
    return


def radial_average(grid, center, pop, binning):
    ind, dist = grid.surrounding_sphere(center, binning.limit)
    discrete = binning.assign(dist)
    counts = fixed_length_bincount(discrete, length=binning.number)
    total = fixed_length_bincount(discrete, length=binning.number, weights=pop[ind])
    return total / counts


def shifted_linspace(start, stop, n):
    """Dividing the range [start:stop] into n bins, returns the bin centers.

    Examples
    --------
    >>> shifted_linspace(0, 6, 3)
    array([1., 3., 5.])
    """
    delta = (stop-start)/n
    return np.linspace(start, stop, n, endpoint=False) + delta/2
