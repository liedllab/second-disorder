#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain
import logging

import mdtraj as md
import numpy as np
import gisttools as gt


class RadialBinning:
    def __init__(self, number, spacing):
        self.number = number
        self.spacing = spacing
        return

    @property
    def shape(self):
        return (self.n_bins,)

    @property
    def centers(self):
        return np.arange(self.number) * self.spacing + self.spacing/2

    @property
    def edges(self):
        return np.arange(self.number+1) * self.spacing

    @property
    def volume(self):
        spheres = self.edges**3 * (4/3 * np.pi)
        return spheres[1:] - spheres[:-1]

    @property
    def limit(self):
        return self.number * self.spacing

    def assign(self, a):
        """Assign distances a to a bin. Are values smaller than *spacing* are
        assigned bin 0, and all values > self.limit are assigned self.number.
        """
        bins = np.intc(a / self.spacing)
        # This is out-of-bounds by one. I keep it this way for
        # consistency with np.digitize(a, self.edges[1:])
        bins = np.clip(bins, 0, self.number)
        return bins

    def coarse_grain(self, factor):
        """Return a new RadialBinning where spacing = self.spacing * factor and
        number = self.number // factor.

        Raises an error if self.number is not divisible by factor."""
        number = self.number // factor
        if number * factor != self.number:
            raise ValueError(f'Cannot coarse grain because self.number is not divisible by {factor}')
        spacing = self.spacing * factor
        return self.__class__(number, spacing)


    def __repr__(self):
        return f"{self.__class__.__name__}({self.number}, {self.spacing})"


class HistogramGrid:
    def __init__(self, hist, occurrences, grid, binning):
        self.occurrences = np.asarray(occurrences).reshape(grid.shape)
        self.hist = hist.reshape(tuple(grid.shape) + (hist.shape[-1],))
        self.grid = grid
        self.binning = binning
        assert binning.number == hist.shape[-1]
        return

    @classmethod
    def from_npz(cls, npz, spacing=None):
        hist = npz['histograms']
        occurrences = npz['occurrences']
        hist[..., 0] = 0
        grid = gt.grid.Grid(npz['grid_origin'], npz['grid_shape'], npz['grid_delta'])
        spacing = npz['spacing']
        binning = RadialBinning(hist.shape[-1], spacing)
        return cls(hist=hist, occurrences=occurrences, grid=grid, binning=binning)

    @classmethod
    def from_npz_name(cls, name, spacing=None):
        with np.load(name) as npz:
            return cls.from_npz(npz, spacing=spacing)

    def save_npz(self, npz):
        np.savez(
            npz,
            histograms=self.hist,
            occurrences=self.occurrences,
            grid_origin=self.grid.origin,
            grid_shape=self.grid.shape,
            grid_delta=self.grid.delta,
            spacing=self.binning.spacing,
        )

    @classmethod
    def empty(cls, grid, binning, dtype=float):
        hist = np.zeros(tuple(grid.shape) + (binning.number,), dtype=dtype)
        occurrences = np.zeros(tuple(grid.shape), dtype=dtype)
        return cls(hist=hist, occurrences=occurrences, grid=grid, binning=binning)

    @property
    def n_bins(self):
        return self.binning.number

    @property
    def shape(self):
        return tuple(self.grid.shape) + (self.n_bins,)

    def flat_occurrences(self):
        """Returns a flat view of self.occurrences.

        Raises an error if this is not possible"""
        out = self.occurrences.view()
        out.shape = (self.grid.size,)
        return out

    def semi_flat_hist(self):
        """Returns a view of self.hist with shape (self.grid.shape, self.n_bins).

        Raises an error if this is not possible"""
        out = self.hist.view()
        out.shape = (self.grid.size, self.n_bins)
        return out

    def rdf(self, refdens=1.):
        with np.errstate(invalid='ignore'):
            rdf = self.hist / self.occurrences[..., np.newaxis]
            rdf[:] = rdf / (self.binning.volume * refdens)
        return rdf

    def coarse_grain(self, factor):
        """Coarse grain along the grid axes, summing all respective voxels.

        Examples
        --------
        >>> binning = RadialBinning(10, .1)
        >>> grid = gt.grid.Grid(origin=-1, shape=20, delta=0.1)
        >>> HistogramGrid.empty(grid, binning).coarse_grain(2).shape
        (10, 10, 10, 10)
        """
        hist = rebin(self.hist, factor, axes=(0, 1, 2))
        occurrences = rebin(self.occurrences, factor, axes=(0, 1, 2))
        grid = self.grid.coarse_grain(factor)
        return self.__class__(hist, occurrences, grid, self.binning)

    def rebin(self, factor):
        """Rebin along the radial axis, and coarse-grain the binning.
        Examples
        --------
        >>> binning = RadialBinning(10, .1)
        >>> grid = gt.grid.Grid(origin=-1, shape=20, delta=0.1)
        >>> HistogramGrid.empty(grid, binning).rebin(2).shape
        (20, 20, 20, 5)
        """
        hist = rebin(self.hist, factor, axes=(3,))
        binning = self.binning.coarse_grain(factor)
        return self.__class__(hist, self.occurrences, self.grid, binning)

    def rescale(self, new_occ):
        """Rescale occurrences AND hist. This does not change the rdf, but can
        be used e.g. as a weighting before coarse graining."""
        reshaped = np.broadcast_to(new_occ, self.occurrences.shape)
        self.hist *= (reshaped / self.occurrences)[..., np.newaxis]
        self.occurrences = reshaped

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"


def rebin(arr, factor, axes=(0,)):
    tempshape = tuple(chain(*(
        (dim//factor, factor) if i in axes else (dim,)
        for i, dim in enumerate(arr.shape)
    )))
    reduce_dims = tuple(a+shift for a, shift in zip(axes, range(1, len(axes)+1)))
    return arr.reshape(tempshape).sum(axis=reduce_dims)


def iter_load(traj_files, top, stride=None):
    for fn in traj_files:
        logging.info(f'Loading trajectory: {fn}')
        for traj in md.iterload(fn, top=top, stride=stride, chunk=1000):
            yield traj


def load_density(fname, rho0):
    gistfile = gt.gist.load_dx(fname, colname='dens')
    gistfile['dens'] = gistfile['dens'] / rho0
    return gistfile


def fixed_length_bincount(arr, length, weights=None):
    out = np.bincount(arr, minlength=length, weights=weights)
    if len(out) > length:
        out = out[:length]
    return out
