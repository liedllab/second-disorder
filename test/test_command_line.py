from collections import namedtuple
import second_disorder.command_line as sd_cmd
import inspect

import pytest


def get_parameter_list(fun):
    params = inspect.signature(fun).parameters
    return params


def parse_input(input_str):
    in_args = input_str.split()
    return sd_cmd.parse_args(in_args)


def args_match_parameter_list(args, params, ignore=()):
    """Check whether the args (as returned by ArgumentParser.parse_args) match
    the parameter names.  The order is not checked.  Elements in ignore will be
    removed from args before checking equality."""
    args_dict = dict(vars(args).items())
    for ig in ignore:
        args_dict.pop(ig)
    return tuple(sorted(args_dict)) == tuple(sorted(params))


ParsingExample = namedtuple('ParsingExample', ['cmd', 'fun'])

@pytest.fixture
def good_parsing_examples():
    return [
        ParsingExample(
            'conditional_density test.parm7 traj1 traj2 --out test.npz',
            sd_cmd.run_conditional_density,
        ),
        ParsingExample(
            'density test.top traj1 traj2 --mol cation --out test.dx',
            sd_cmd.run_density,
        ),
        ParsingExample(
            'density_shells dens.dx --out test.dx --hist_spacing .3',
            sd_cmd.run_density_shells,
        ),
        ParsingExample(
            'entropy hist.npz ref_hist2.npz dens1.dx --entropy_out ent.dx --ksa_out ksa.dx --rho0_1 4.0 --rho0_2 0.4  --bulk_hist bulk_hist.csv',
            sd_cmd.run_entropy,
        ),
        ParsingExample(
            'add_histograms water-water-1.npz water-water-2.npz --out water-water-combined.npz',
            sd_cmd.run_add_histograms,
        ),
        ParsingExample(
            'trans_entropy -p top.parm7 -x traj1.nc traj2.nc --ions NA CL --origin -2 -2 -2 --delta 0.1 0.1 0.1 --shape 40 40 40 --rho0 0.06 0.07',
            sd_cmd.run_trans_entropy,
        )
    ]


def test_empy_command_line_exits():
    with pytest.raises(SystemExit):
        parse_input('')


def test_parsers_produce_correct_functions(good_parsing_examples):
    for ex in good_parsing_examples:
        args = parse_input(ex.cmd)
        assert args.func == ex.fun


def test_parser_outputs_match_function_signatures(good_parsing_examples):
    for ex in good_parsing_examples:
        args = parse_input(ex.cmd)
        params = get_parameter_list(ex.fun)
        assert args_match_parameter_list(args, params, ignore=['func', 'verbose'])


def test_density_parsing_with_typo_in_mol():
    with pytest.raises(SystemExit):
        typo = 'density test.top traj1 traj2 --mol cccation --out test.dx'
        parse_input(typo)
