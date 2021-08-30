#!/usr/bin/env python3
import numpy as np
import gisttools as gt
import mdtraj as md


from second_disorder.base import iter_load


def run_density(top, trajfiles, mol, stride, out):
    top = md.load_topology(top)
    selections = {'cation': 'name NA', 'anion': 'name CL', 'water': 'name O and resname HOH'}
    atoms = top.select(selections[mol])
    frame1 = md.load_frame(trajfiles[0], 0, top=top, atom_indices=top.select('resid 0'))
    center = np.mean(frame1.xyz, axis=1)[:, np.newaxis, :]
    grid = gt.grid.Grid.centered(0, 160, .025)
    population = np.zeros(grid.size, dtype=int)
    n_frames = 0
    for traj in iter_load(traj_files=trajfiles, top=top, stride=stride):
        traj = traj.atom_slice(atoms)
        traj.xyz -= center
        for frame_crd in traj.xyz:
            n_frames += 1
            voxels = grid.assign(frame_crd)
            # Assumes that the grid is sufficiently fine, i.e., there are never 2 molecules in the same voxel at the same time.
            population[voxels[voxels != -1]] += 1
    population = population / n_frames
    grid.save_dx(population, out, 'population')
