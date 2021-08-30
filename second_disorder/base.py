#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain

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
        bins = np.intc(a / self.spacing)
        # This is out-of-bounds by one. I keep it this way for
        # consistency with np.digitize(a, self.edges[1:])
        bins = np.clip(bins, 0, self.number)
        return bins

    def coarse_grain(self, factor):
        number = self.number // factor
        if number * factor != self.number:
            raise ValueError(f'Cannot coarse grain because self.number is not divisible by {factor}')
        spacing = self.spacing * factor
        return self.__class__(number, spacing)


    def __repr__(self):
        return f"{self.__class__.__name__}({self.number}, {self.spacing})"


class HistogramGrid:
    def __init__(self, hist, central, grid, binning):
        self.central = np.asarray(central).reshape(grid.shape)
        self.hist = hist.reshape(tuple(grid.shape) + (hist.shape[-1],))
        self.grid = grid
        self.binning = binning
        assert binning.number == hist.shape[-1]
        return

    @classmethod
    def from_npz(cls, npz, spacing=None):
        hist = npz['histograms']
        central = np.copy(hist[..., 0])
        hist[..., 0] = 0
        grid = gt.grid.Grid(npz['grid_origin'], npz['grid_shape'], npz['grid_delta'])
        if spacing is None:
            spacing = npz['spacing']
        binning = RadialBinning(hist.shape[-1], spacing)
        return cls(hist=hist, central=central, grid=grid, binning=binning)

    @classmethod
    def from_npz_name(cls, name, spacing=None):
        with np.load(name) as npz:
            return cls.from_npz(npz, spacing=spacing)

    def save_npz(self, npz):
        histograms = np.copy(self.hist).reshape(-1, self.n_bins)
        histograms[:, 0] = self.central.ravel()
        np.savez(
            npz,
            histograms=histograms,
            grid_origin=self.grid.origin,
            grid_shape=self.grid.shape,
            grid_delta=self.grid.delta,
            spacing=self.binning.spacing,
        )

    @classmethod
    def empty(cls, grid, binning, dtype=float):
        hist = np.zeros(tuple(grid.shape) + (binning.number,), dtype=dtype)
        central = np.zeros(tuple(grid.shape), dtype=dtype)
        return cls(hist=hist, central=central, grid=grid, binning=binning)

    @property
    def n_bins(self):
        return self.binning.number

    @property
    def shape(self):
        return tuple(self.grid.shape) + (self.n_bins,)

    def flat_central(self):
        """Returns a flat view of self.central.

        Raises an error if this is not possible"""
        out = self.central.view()
        out.shape = (self.grid.size,)
        return out

    def semi_flat_hist(self):
        """Returns a view of self.hist with shape (self.grid.shape, self.n_bins).

        Raises an error if this is not possible"""
        out = self.hist.view()
        out.shape = (self.grid.size, self.n_bins)
        return out

    def rdf(self, refdens=1.):
        return self.reference(self.binning.volume * refdens)

    def reference(self, dens):
        with np.errstate(invalid='ignore'):
            rdf = self.hist / self.central[..., np.newaxis]
            rdf[:] = rdf / dens
        return rdf

    def coarse_grain(self, factor):
        hist = rebin(self.hist, factor, axes=(0, 1, 2))
        central = rebin(self.central, factor, axes=(0, 1, 2))
        grid = self.grid.coarse_grain(factor)
        return self.__class__(hist, central, grid, self.binning)

    def rebin(self, factor):
        hist = rebin(self.hist, factor, axes=(3,))
        binning = self.binning.coarse_grain(factor)
        return self.__class__(hist, self.central, self.grid, binning)

    def coarse_grain_mean(self, factor, weights=None):
        if weights is None:
            weights = np.ones(tuple(self.grid.shape))
        weights = np.asarray(weights).reshape(self.grid.shape)
        total = rebin(self.hist * weights[..., np.newaxis], factor, axes=(0, 1, 2))
        weight_sum = rebin(weights, factor, axes=(0, 1, 2))
        hist = total / weight_sum[..., np.newaxis]
        central = rebin(self.central, factor, axes=(0, 1, 2))
        grid = self.grid.coarse_grain(factor)
        return self.__class__(hist, central, grid, self.binning)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"


def rebin(arr, factor, axes=(0,)):
    tempshape = tuple(chain(*(
        (dim//factor, factor) if i in axes else (dim,)
        for i, dim in enumerate(arr.shape)
    )))
    reduce_dims = tuple(a+shift for a, shift in zip(axes, range(1, len(axes)+1)))
    return arr.reshape(tempshape).sum(axis=reduce_dims)


