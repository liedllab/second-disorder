#!/usr/bin/env python3

import logging

import numpy as np
import gisttools as gt
import mdtraj as md
import math
import numba
from scipy.spatial import cKDTree
from second_disorder.base import HistogramGrid, RadialBinning, iter_load


def run_conditional_density(top, trajfiles, mol1, mol2, stride, out, n_bins, bin_spacing, grid_spacing, grid_bins):
    top = md.load_topology(top)
    frame1 = md.load_frame(trajfiles[0], 0, top=top, atom_indices=top.select('resid 0'))
    center = np.mean(frame1.xyz, axis=1)[:, np.newaxis, :]
    selections = {'cation': 'name NA', 'anion': 'name CL', 'water': 'name O and resname HOH'}
    sel1 = top.select(selections[mol1])
    sel2 = top.select(selections[mol2])
    grid = gt.grid.Grid.centered(0, grid_bins, grid_spacing)
    binning = RadialBinning(n_bins, bin_spacing)
    histograms = HistogramGrid.empty(grid, binning, dtype=int)
    for traj in iter_load(trajfiles, top=top, stride=stride):
        traj.xyz -= center
        a = traj.atom_slice(sel1)
        b = traj.atom_slice(sel2)
        insert_to_histograms(grid, a, b, histograms)
    histograms.save_npz(out)


def insert_to_histograms(grid, traj_a, traj_b, histograms):
    assert traj_a.n_frames == traj_b.n_frames
    for a, b, cell_lengths in zip(traj_a.xyz, traj_b.xyz, traj_a.unitcell_lengths):
        _insert_to_histograms(grid, a, b, cell_lengths, histograms)


def _insert_to_histograms(grid, a, b, cell_lengths, histograms):
    binning = histograms.binning
    b = b % cell_lengths
    b[b == cell_lengths] = 0  # Needed when a value is exactly on the boundary
    tree_b = cKDTree(b, boxsize=cell_lengths)
    voxels = grid.assign(a)
    in_box = voxels != -1
    hist = histograms.semi_flat_hist()
    occurrences = histograms.flat_occurrences()
    for x1, vox, nbrs in zip(a[in_box], voxels[in_box], tree_b.query_ball_point(a[in_box], binning.limit)):
        _insert_to_histograms_inner(b[nbrs], x1, cell_lengths, binning.number, binning.spacing, hist[vox])
        occurrences[vox] += 1


@numba.njit
def _insert_to_histograms_inner(points, ref_point, cell_lengths, n_bins, bin_spacing, hist):
    assert len(points.shape) == 2 and points.shape[1] == 3
    assert ref_point.shape == (3,)
    l1, l2, l3 = cell_lengths
    l1_half, l2_half, l3_half = l1 / 2, l2 / 2, l3 / 2
    x_ref, y_ref, z_ref = ref_point
    for (x, y, z) in points:
        dx = ((x - x_ref) + l1_half) % l1 - l1_half
        dy = ((y - y_ref) + l2_half) % l2 - l2_half
        dz = ((z - z_ref) + l3_half) % l3 - l3_half
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        bin = int(dist / bin_spacing)
        if bin < n_bins:
            hist[bin] += 1
