#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
from gisttools.grid import Grid
# from scipy.spatial import cKDTree
from pykdtree.kdtree import KDTree as cKDTree
import mdtraj as md

from second_disorder.base import fixed_length_bincount, iter_load

EULER_MASCHERONI = 0.5772156649015329
# Exact since the re-definition of the Kelvin in 2019
BOLTZMANN_CONSTANT = 1.380649E-23


def run_trans_entropy(parm, trajs, ions, origin, delta, shape, outprefix, rho0):
    assert len(ions) == len(rho0), "Need one reference density for each ion"
    TEMPERATURE = 300
    ion_coords = load_ion_coordinates_angstrom(
        parm,
        trajs,
        ions,
        center_sel='resid 0',
    )
    ion_rho0 = dict(zip(ions, rho0))
    grid = Grid(origin, shape, delta)
    for ion in ions:
        logging.info(f"Processing ion: {ion}")
        write_ion_density(
            outfile=f"{outprefix}{ion}.dx",
            coords=ion_coords[ion],
            grid=grid,
            refdens=ion_rho0[ion],
            temperature=TEMPERATURE,
        )


def load_ion_coordinates_angstrom(parm, trajs, ions, center_sel):
    top = md.load_topology(parm)
    solute = top.select(center_sel)
    logging.info(f'Centering selection {center_sel}, with atoms {solute}')
    coords = {ion: [] for ion in ions}
    atoms = {ion: select_ion(top, ion) for ion in ions}
    for traj in iter_load(trajs, top=top):
        center(traj, solute)
        for ion in ions:
            # convert to Angstrom
            coords[ion].append(traj.xyz[:, atoms[ion], :] * 10)
    concat = {ion: np.concatenate(coords[ion]) for ion in ions}
    return concat


def write_ion_density(outfile, coords, grid, refdens, temperature):
    out = strans_dens(coords, grid, refdens, temperature)
    grid.save_dx(out, outfile, colname='TStrans_dens')


def strans_dens(coords, grid, rho0, temperature):
    """Calculate the free energy contribution (in kcal/mol) of STrans_dens
    given atomic coordinates, a Grid, and rho0.
    
    Reference:
    J. Chem. Theory Comput. 2019, 15, 5872−5882, Equation 6"""
    assert len(coords.shape) == 3
    n_frames, n_atoms, n_dims = coords.shape
    if n_atoms == 0:
        return np.zeros(grid.size, dtype=np.float32)
    flat = coords.reshape(n_frames*n_atoms, n_dims)
    entropy = trans_entropy(flat, grid, rho0 * n_frames) / n_frames
    s_trans_dens = entropy * (temperature / grid.voxel_volume)
    return s_trans_dens


def trans_entropy(coords, grid, density):
    """Entropy of a point cloud with given reference density, localized to a grid."""
    coords = np.atleast_2d(coords)
    assert len(coords.shape) == 2 and coords.shape[-1] == 3
    indices = grid.assign(coords)
    in_grid = np.flatnonzero(indices != -1)
    indices = indices[in_grid]
    coords = np.ascontiguousarray(coords[in_grid], dtype=np.float32)
    # The sum is performed as a weighted bincount
    ln_gnn_sums = fixed_length_bincount(
        indices,
        length=grid.size,
        weights=log_dens(coords, refdens=density),
    )
    Nk = fixed_length_bincount(indices, length=grid.size)
    entropy = BOLTZMANN_CONSTANT * Nk * (ln_gnn_sums / Nk + EULER_MASCHERONI)
    entropy[Nk == 0] = 0.
    return entropy


def log_dens(coords, refdens):
    """Negative logarithm of the density at each point, relative to refdens"""
    d_trans = nn_distances(coords)
    return np.log(sphere_volume(d_trans) * refdens)


def nn_distances(coords):
    """Return the distance between every point in coords and its nearest neighbor.
    
    Parameters
    ----------
    coords: np.ndarray, shape=(n_points, 3)
    """
    assert len(coords.shape) == 2 and coords.shape[1] == 3
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=2)
    dist = dist[:, 1]
    return dist


def sphere_volume(radius):
    return (radius ** 3) * (4 * np.pi / 3)


def center(traj: md.Trajectory, atoms: np.ndarray):
    com = traj.xyz[:, atoms, :].mean(1)
    traj.xyz -= com[:, np.newaxis, :]
    return


def select_ion(top, ion):
    """Try to select an ion using mdtraj. Returns an empty array if the selection fails."""
    name = 'N' if ion == 'NH4' else ion
    atoms = top.select(f"name {name} and resname {ion}")
    return np.int64(atoms)
