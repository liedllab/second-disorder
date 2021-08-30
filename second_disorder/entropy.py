#!/usr/bin/env python3
import gisttools as gt
import numpy as np
import pandas as pd
from second_disorder.base import HistogramGrid, RadialBinning, rebin, load_density
import logging

def run_entropy(hist, ref_hist, dens, entropy_out, ksa_out, rebin, rho0_1, rho0_2, bulk_hist, coarse_grain, kT):
    hist = HistogramGrid.from_npz_name(hist).coarse_grain(coarse_grain).rebin(rebin)
    binning = hist.binning
    grid = hist.grid
    ref_hist = HistogramGrid.from_npz_name(ref_hist)
    rho0_1 = rho0_1 * 1000
    rho0_2 = rho0_2 * 1000
    logging.info(f'{rho0_1=}, {rho0_2=}')
    fine_dens = load_density_arr(dens, rho0_1)
    dens = rebin_3d_array_to(fine_dens, tuple(grid.shape)).ravel()
    ref_hist = coarse_grain_histgrid_to(ref_hist, grid, weights=fine_dens)
    ref_hist = ref_hist.rebin(ref_hist.n_bins // binning.number)
    # ref_hist.occurrences[:] = 1  # After coarse graining!
    g0 = coarse_grain_rdf(load_rdf(bulk_hist), binning)['g'].values
    entropy = np.zeros(grid.size, dtype=float)
    ksa_entropy = np.zeros(grid.size, dtype=float)
    rdfs = hist.rdf(rho0_2).reshape(grid.size, binning.number)
    ref_rdfs = ref_hist.rdf(rho0_2).reshape(grid.size, binning.number)

    logging.info(f'{g0.shape=}')
    logging.info(g0)
    logging.info(f'{rdfs.shape=}, printing first line')
    logging.info(rdfs[0])
    logging.info(f'{ref_rdfs.shape=}, printing first line')
    logging.info(ref_rdfs[0])
    logging.info(f'{dens.shape=}, printing first line')
    logging.info(dens[0])
    for i_voxel in range(grid.size):
        if i_voxel % 1000 == 0:
            print(f'{i_voxel/grid.size:.02f}')
        Gsv = dens[i_voxel]
        Gsvp = ref_rdfs[i_voxel]
        ginh = rdfs[i_voxel] / Gsvp
        # The factor of 1000 is because we scale the grid from nm to Angstrom
        # before writing it out. 1 kcal/A^3 = 1000 kcal/nm^3
        prefactor = -1/2 * kT * rho0_1 * rho0_2 * Gsv / 1000
        entropy[i_voxel] = prefactor * ent2_inner_integral(binning, Gsvp, ginh, g0)
        ksa_entropy[i_voxel] = prefactor * ent2_inner_integral(binning, Gsvp, g0, g0)
        if i_voxel == 0:
            logging.info(f'{prefactor=}')
            logging.info(f'{entropy[i_voxel]=}')
            logging.info(f'{ksa_entropy[i_voxel]=}')
    grid.scale(10.).save_dx(entropy, entropy_out, colname='2nd_order_entropy')
    grid.scale(10.).save_dx(ksa_entropy, ksa_out, colname='ksa_entropy')
    return


def rebin_3d_array_to(a, shape):
    factor = a.shape[0] // shape[0]
    assert np.allclose(a.shape, np.array(shape) * factor)
    return rebin(a, factor, axes=(0, 1, 2)) / factor**3


def coarse_grain_histgrid_to(histgrid, grid: gt.grid.Grid, weights: np.ndarray):
    """This modifies the input histgrid!!! Returns a new HistogramGrid coarse
    grained to the shape of the given Grid, using the given weights."""
    factor = histgrid.grid.shape[0] // grid.shape[0]
    assert np.allclose(grid.shape * factor, histgrid.grid.shape)
    histgrid.rescale(weights.reshape(histgrid.grid.shape))
    return histgrid.coarse_grain(factor)


def load_density_arr(fname, rho0):
    dens = load_density(fname, rho0)
    return dens['dens'].values.reshape(dens.grid.shape)

def entropy_contrib(a):
    with np.errstate(invalid='ignore', divide='ignore'):
        log_a = np.log(a)
    log_a[a == 0] = 0
    return a * log_a - a + 1


def ent2_inner_integral(binning, Gsv, ginh, g0):
    vals = ent2_inner(binning, Gsv, ginh, g0)
    return np.sum(vals[~np.isnan(vals)])


def ent2_inner(binning, Gsv, ginh, g0):
    return binning.volume * (
        (Gsv * entropy_contrib(ginh))
        - (entropy_contrib(g0))
    )


def load_rdf(fname):
    rdf = pd.read_csv(fname, names=['r', 'g'], skiprows=1, sep='\s+')
    rdf['r'] = rdf['r'] / 10.
    return rdf


def coarse_grain_rdf(rdf, binning):
    # TODO: should raise an error if the bin edges don't match.
    centers = rdf['r']
    fine = RadialBinning(len(centers), centers[1]-centers[0])
    rdf = rdf.assign(dv=fine.volume, bin=binning.assign(centers))
    cg = (
        rdf
        .eval('n = g * dv')
        [['dv', 'n', 'bin']]
        .groupby('bin')
        .sum()
        .eval('g = n / dv')
        .drop(index=binning.number)
    )
    cg['r'] = RadialBinning(cg.shape[0], binning.spacing).centers
    return cg


def shifted_linspace(start, stop, n):
    delta = (stop-start)/n
    return np.linspace(start, stop, n, endpoint=False) + delta/2
