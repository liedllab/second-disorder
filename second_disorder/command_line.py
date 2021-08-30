import argparse
import logging
import sys

from second_disorder.conditional_density import run_conditional_density
from second_disorder.density import run_density
from second_disorder.density_shells import run_density_shells
from second_disorder.entropy import run_entropy
from second_disorder.add_histograms import run_add_histograms
from second_disorder.trans_entropy import run_trans_entropy


def main():
    args = parse_args(sys.argv[1:])
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    func = args.func
    delattr(args, 'func')
    delattr(args, 'verbose')
    func(**vars(args))


def parse_args(arg_strings):
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    subparsers = parser.add_subparsers()

    cond_dens_parser = subparsers.add_parser('conditional_density')
    cond_dens_parser.set_defaults(func=run_conditional_density)
    cond_dens_parser.add_argument('top')
    cond_dens_parser.add_argument('trajfiles', nargs='+')
    cond_dens_parser.add_argument('--mol1', choices=('cation', 'anion', 'water'))
    cond_dens_parser.add_argument('--mol2', choices=('cation', 'anion', 'water'))
    cond_dens_parser.add_argument('--stride', type=int, default=1)
    cond_dens_parser.add_argument('--out', required=True)
    cond_dens_parser.add_argument('--n_bins', default=40, type=int)
    cond_dens_parser.add_argument('--bin_spacing', default=.025, type=float)
    cond_dens_parser.add_argument('--grid_spacing', default=.1, type=float)
    cond_dens_parser.add_argument('--grid_bins', default=40, type=int)

    density_parser = subparsers.add_parser('density')
    density_parser.set_defaults(func=run_density)
    density_parser.add_argument('top')
    density_parser.add_argument('trajfiles', nargs='+')
    density_parser.add_argument('--mol', choices=('cation', 'anion', 'water'))
    density_parser.add_argument('--stride', type=int, default=1)
    density_parser.add_argument('--out', required=True)

    density_shells_parser = subparsers.add_parser('density_shells')
    density_shells_parser.set_defaults(func=run_density_shells)
    density_shells_parser.add_argument('dens2')
    density_shells_parser.add_argument('--out', required=True)
    density_shells_parser.add_argument('--hist_spacing', default=.0125, type=float)
    density_shells_parser.add_argument('--hist_bins', default=80, type=int)

    entropy_parser = subparsers.add_parser('entropy')
    entropy_parser.set_defaults(func=run_entropy)
    entropy_parser.add_argument('hist')
    entropy_parser.add_argument('ref_hist')
    entropy_parser.add_argument('dens')
    entropy_parser.add_argument('--entropy_out', required=True)
    entropy_parser.add_argument('--ksa_out', required=True)
    entropy_parser.add_argument('--rebin', type=int, default=1)
    entropy_parser.add_argument('--rho0_1', type=float, required=True)
    entropy_parser.add_argument('--rho0_2', type=float, required=True)
    entropy_parser.add_argument('--bulk_hist', required=True, type=str)
    entropy_parser.add_argument('--coarse_grain', type=int, default=1)
    entropy_parser.add_argument('--kT', type=float, default=0.59616) # 300K

    add_histograms_parser = subparsers.add_parser('add_histograms')
    add_histograms_parser.set_defaults(func=run_add_histograms)
    add_histograms_parser.add_argument('grids', nargs='+')
    add_histograms_parser.add_argument('--out', required=True)

    trans_entropy_parser = subparsers.add_parser('trans_entropy')
    trans_entropy_parser.set_defaults(func=run_trans_entropy)
    trans_entropy_parser.add_argument("-p", "--parm", required=True, help="Topology that can be read by mdtraj.")
    trans_entropy_parser.add_argument("-x", "--trajs", nargs="+", required=True, help="Trajectory that can be read by mdtraj.")
    trans_entropy_parser.add_argument("--ions", nargs="+", required=True)
    trans_entropy_parser.add_argument("--origin", nargs=3, type=float, help="Grid origin")
    trans_entropy_parser.add_argument("--delta", nargs=3, type=float, help="Grid delta")
    trans_entropy_parser.add_argument("--shape", nargs=3, type=int, help="Grid shape")
    trans_entropy_parser.add_argument("--outprefix", default="dTStrans-")
    trans_entropy_parser.add_argument("--strip_wat", action='store_true')

    args = parser.parse_args(arg_strings)
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit('For help regarding a subcommand use second_disorder {subcommand} -h.')
    return args

if __name__ == "__main__":
    main()
