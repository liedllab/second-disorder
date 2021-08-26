#!/usr/bin/env python3

import logging

import numpy as np
import gisttools as gt
import mdtraj as md
import math
import numba
from scipy.spatial import cKDTree
from base import HistogramGrid, RadialBinning

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('top')
    parser.add_argument('trajfiles', nargs='+')
    parser.add_argument('--mol1', choices=('cation', 'anion', 'water'))
    parser.add_argument('--mol2', choices=('cation', 'anion', 'water'))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n_bins', default=40, type=int)
    parser.add_argument('--bin_spacing', default=.025, type=float)
    parser.add_argument('--grid_spacing', default=.1, type=float)
    parser.add_argument('--grid_bins', default=40, type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    top = md.load_topology(args.top)
    frame1 = md.load_frame(args.trajfiles[0], 0, top=top, atom_indices=top.select('resid 0'))
    center = np.mean(frame1.xyz, axis=1)[:, np.newaxis, :]
    selections = {'cation': 'name NA', 'anion': 'name CL', 'water': 'name O and resname HOH'}
    sel1 = top.select(selections[args.mol1])
    sel2 = top.select(selections[args.mol2])
    grid = gt.grid.Grid.centered(0, args.grid_bins, args.grid_spacing)
    binning = RadialBinning(args.n_bins, args.bin_spacing)
    histograms = HistogramGrid.empty(grid, binning, dtype=int)
    for traj in iter_load(args.trajfiles, top=top, stride=args.stride, offset=center):
        a = traj.atom_slice(sel1)
        b = traj.atom_slice(sel2)
        insert_to_histograms(grid, a, b, histograms)
    histograms.save_npz(args.out)


def iter_load(traj_files, top, stride=None, offset=0):
    for fn in traj_files:
        logging.info(f'Loading trajectory: {fn}')
        # traj = md.load(fn, top=top, stride=stride)
        for traj in md.iterload(fn, top=top, stride=stride, chunk=1000):
            traj.xyz -= offset
            yield traj


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
    central = histograms.flat_central()
    for x1, vox, nbrs in zip(a[in_box], voxels[in_box], tree_b.query_ball_point(a[in_box], binning.limit)):
        _insert_to_histograms_inner(b[nbrs], x1, cell_lengths, binning.number, binning.spacing, hist[vox])
        central[vox] += 1


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


if __name__ == '__main__':
    main()
