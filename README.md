# entropy
This is a Python implementation of 1st- and 2nd- order entropies for GIST (Grid Inhomogeneous Solvation Theory). Specifically, this can be used to compute the following quantities:

* First-order translational entropy. This is slower than the versions in cpptraj or GIGIST, and meant for ions in a salt-water mixture. The advantage is that it does not rely on the water density being relatively high compared to the grid spacing, so it should be used for ions.
* Second-order entropy, based on a representation of conditional 1D histograms on a 3D grid. This is a less detailed representation than in the work by Gilson and coworkers (https://dx.doi.org/10.1021%2Facs.jctc.5b00939). On the other hand, it is easier to obtain converged results, especially for low-density components like ions.
* The same scripts can be used to compute the second-order entropy using the Kirkwood superposition approximation. At the moment, both are computed simultaneously, but in future versions it will be possible to do those computations separately.

## Installation
The second-order scripts should all be in the same directory. There is no installation via setup.py (yet), so just put them into some folder. Many of the scripts depend on the gisttools, so install that first (https://github.com/liedllab/gisttools). The dependencies of gisttools are also used here. Additionally, the trans-entropy script currently uses the pykdtree module. If you don't want to install pykdtree, you can use the (slower) cKDTree implementation from scipy instead. Just replace the import statement.

## Usage
The second-order scripts need to be used together to compute a second-order entropy. The workflow is like this:
* histograms.py: Compute conditional histograms *g_inh(nu, nu_prime, r, distance)* from a trajectory.
* density.py: Compute the molecular density from a trajectory. The output is a .dx file that corresponds to the "*population*" in a regular GIST run.
* ref_histograms.py: Compute the reference histograms *G(nu_prime, r, distance)* from the .dx file output by density.py.
* entropy.py: Compute the second-order entropy from the output of the other scripts. To obtain e.g. the water-cation entropy, you need the water-cation conditional histogram from histograms.py, the water density from density.py, the cation reference histogram from ref_histograms.py, and the water-cation bulk rdf (can be obtained from cpptraj. Be careful to use a grid spacing that is equal to, or an integer fraction of, the grid spacing in the conditional histogram). Furthermore, you need the reference densities (as in regular GIST), and the number of frames that were used to compute the density (the .dx file).

In the above, *nu* and *nu_prime* refer to molecular species (like water and cation), *r* is the position of a grid voxel, and *distance* is distance to such a grid voxel.

## Note
This is the version of the scripts used for the article "Explicit Solvation Thermodynamics in Ionic Solution - An Extension of Grid Inhomogeneous Solvation Theory Computes Solvation Free Energy in Salt-Water mixtures." I plan to rework the scripts, since they are currently poorly tested and documented, and the information flow (normalization of histograms etc.) is a bit inconsistent. The rework will happen in the development branch, so if you are curious, check it out.
