#!/usr/bin/env python3
import numpy as np
import gisttools as gt
import mdtraj as md


def run_density(top, trajfiles, mol, stride, out):
    top = md.load_topology(top)
    na = top.select('name NA')
    cl = top.select('name CL')
    wat = top.select('name O and resname HOH')
    selections = {'cation': na, 'anion': cl, 'water': wat}
    frame1 = md.load_frame(trajfiles[0], 0, top=top, atom_indices=top.select('resid 0'))
    center = np.mean(frame1.xyz, axis=1)[:, np.newaxis, :]
    grid = gt.grid.Grid.centered(0, 160, .025)
    population = np.zeros(grid.size, dtype=int)
    trajs = iter_load(
        atoms=selections[mol],
        traj_files=trajfiles,
        top=top,
        stride=stride,
        offset=center,
    )
    for traj in trajs:
        for frame_crd in traj.xyz:
            voxels = grid.assign(frame_crd)
            # Assumes that the grid is sufficiently fine, i.e., there are never 2 molecules in the same voxel at the same time.
            population[voxels[voxels != -1]] += 1
    grid.save_dx(population, out, 'population')


def iter_load(atoms, traj_files, top, stride=None, offset=0):
    for fn in traj_files:
        for traj in md.iterload(fn, top=top, atom_indices=atoms, stride=stride, chunk=1000):
            traj.xyz -= offset
            yield traj
