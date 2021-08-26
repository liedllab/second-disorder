#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from gisttools.grid import Grid
from gisttools.gist import Gist
# from scipy.spatial import cKDTree
from pykdtree.kdtree import KDTree as cKDTree
import mdtraj as md
from typing import Tuple

def dtrans(coords):
    """Given a list of coordinates of the same species, calculate dtrans.
    
    See Ref. 1, Eq. 11"""
    # coords = np.asarray(coords).reshape(-1, 3)
    assert len(coords.shape) == 2 and coords.shape[1] == 3
    print('Constructing Tree')
    tree = cKDTree(coords)
    print('Querying Tree')
    dist, _ = tree.query(coords, k=2)
    dist = dist[:, 1]
    return dist

def vtrans(coords):
    """Given a list of coordinates of the same species, calculate Vtrans.
    
    See Ref. 1, Eq. 9"""
    return dtrans(coords) ** 3 * (4 * np.pi / 3)

def gnn(coords, rho0):
    """Given a list of coordinates of the same species, calculate g_NN(r).
    
    See Ref. 1, Eq. 7"""
    return 1. / (vtrans(coords) * rho0)

def ln_gnn(coords, rho0, n_frames):
    return np.log(vtrans(coords) * rho0 * n_frames)

def strans_dens(coords, grid, rho0, n_frames, kT=0.5961612):
    """Calculate STrans_dens given atomic coordinates, an Grid, and rho0.
    
    See Ref. 1, Eq. 5"""
    GAMMA = 0.5772156649
    coords = np.atleast_2d(coords)
    assert coords.shape[-1] == 3
    coords = coords.reshape(-1, 3)
    
    indices = grid.assign(coords)
    in_grid = np.flatnonzero(indices != -1)
    indices = indices[in_grid]
    coords = np.ascontiguousarray(coords[in_grid], dtype=np.float32)
    # The sum is performed as a weighted bincount
    ln_gnn_sums = np.bincount(
        indices,
        weights=ln_gnn(coords, rho0=rho0, n_frames=n_frames),
        minlength=grid.size
    )
    Nk = np.bincount(indices, minlength=grid.size)
    s_trans = kT * (ln_gnn_sums / Nk + GAMMA)
    s_trans_dens = s_trans * Nk / (grid.voxel_volume * n_frames)
    s_trans_dens[~np.isfinite(s_trans_dens)] = 0.
    print(s_trans_dens.shape, s_trans_dens.size)
    return s_trans_dens

def sliced_by_selection(traj: md.Trajectory, sel: str):
    atoms = np.atleast_1d(traj.top.select(sel))
    if atoms.size == 0:
        atoms = []
    return traj.atom_slice(atoms)

def convert_traj_units(traj: md.Trajectory, factor: float = 10.):
    return md.Trajectory(
        traj.xyz * factor,
        traj.top,
        traj.time,
        traj.unitcell_lengths * factor if traj.unitcell_lengths is not None else None,
        traj.unitcell_angles
    )

def center(traj: md.Trajectory, sel='resid 0'):
    atoms = traj.top.select(sel)
    print(atoms)
    com = traj.xyz[:, atoms, :].mean(1)
    traj.xyz -= com[:, np.newaxis, :]
    return

def rho0(
    centers: np.ndarray,
    ref_coords: np.ndarray,
    grid: Grid,
    dlim: Tuple[float, float],
    n_frames: int
) -> float:
    """Calculate reference density from a binning of centers to grid voxels, in
    a distance within dlim."""
    indices = grid.assign(centers)
    valid = np.flatnonzero(indices != -1)
    population = np.bincount(indices[valid], minlength=grid.size)
    gf = Gist(pd.DataFrame({'pop': population}), grid=grid)
    bins, (ion_rdf, vox_rdf) = gf.multiple_rdfs(
        ['pop', 'voxels'],
        centers=ref_coords,
        rmax=dlim[1],
        bins=100,
        col_suffix=''
    )
    ion_rdf /= grid.voxel_volume  # should not be density weighted
    vox_rdf *= grid.voxel_volume  # should be density weighted but isn't...
    rho0 = ion_rdf[bins > dlim[0]].sum() / vox_rdf[bins > dlim[0]].sum() / n_frames
    return rho0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parm", required=True, help="Topology that can be read by mdtraj.")
    parser.add_argument("-x", "--trajs", nargs="+", required=True, help="Trajectory that can be read by mdtraj.")
    parser.add_argument("--ions", nargs="+", required=True)
    parser.add_argument("--origin", nargs=3, type=float, help="Grid origin")
    parser.add_argument("--delta", nargs=3, type=float, help="Grid delta")
    parser.add_argument("--shape", nargs=3, type=int, help="Grid shape")
    parser.add_argument("--outprefix", default="dTStrans-")
    parser.add_argument("--strip_wat", action='store_true')
    args = parser.parse_args()
    top = md.load_topology(args.parm)
    if args.strip_wat:
        sel = top.select('not resname HOH')
        print(str(len(sel)) + " atoms left after stripping water.")
        traj = md.join([md.load(t, top=top, atom_indices=sel) for t in args.trajs])
        top = traj.top
    else:
        traj = md.join([md.load(t, top=top) for t in args.trajs])
    center(traj)
    grid = Grid(args.origin, args.shape, args.delta)
    for ion in args.ions:
        print(ion)
        write_ion_density(ion, grid, traj, args.outprefix)


def write_ion_density(ion_name, grid, traj, prefix):
    if ion_name == 'NH4':
        name = 'N'
    else:
        name = ion_name
    ion_traj = convert_traj_units(
        sliced_by_selection(traj, f"name {name} and resname {ion_name}"),
        10.
    )
    ion_rho0 = rho0(
        ion_traj.xyz.reshape(-1, 3), 
        sliced_by_selection(traj[0], 'resid 0').xyz[0],
        grid,
        (12, 20),
        traj.n_frames
    )
    print(f'{ion_name} rho0: {ion_rho0}')
    if ion_traj.n_atoms != 0:
        s_dens = strans_dens(ion_traj.xyz, grid, ion_rho0, traj.n_frames)
    else:
        s_dens = np.zeros(grid.size, dtype=np.float32)
    outname = f"{prefix}{ion_name}.dx"
    grid.save_dx(s_dens, outname, ion_name)


if __name__ == "__main__":
    main()
