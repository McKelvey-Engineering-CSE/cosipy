from argparse import ArgumentParser
from pathlib import Path

# build-time defaults

model_dir = Path("/home/jbuhler/adapt")
training_stats_dir = Path("queries")

default_bkg_model_path = model_dir / "adapt_bkg_model.h5"

default_response_path = model_dir / "adapt_response_w_area.h5"

default_mapping_times_path = training_stats_dir / "short_mapping_time_stats.csv"
default_success_probs_path = training_stats_dir / "short_5.36x4.5_tiling.csv"


def parse_args(argv):
    """
    Parse command-line arguments to mapping script.

    Parameters
    ----------
    argv : contents of sys.argv

    Returns
    -------
    dict of argument key / value pairs.  See long argument names below
    for information on keys, types, and default values.  NB: all
    dashes in long argument names get transformed to underscores in
    the dictionary keys.

    """

    parser = ArgumentParser(description="Likelihood mapper for ADAPT transients",
                            epilog='Valid values for endpoint_mode include a numerical deadline in secs (implies utility method) or one of "nodeadline" (deadline-oblivious method) or "gt" (use actual transient length).')

    # positional arguments
    parser.add_argument('transient_path', type=Path,
                        help='path from which to read transients')

    parser.add_argument('endpoint_mode',
                        help='deadline/method for endpoint detection')

    # keyword arguments
    parser.add_argument('-e', '--endpoint-resolution', type=float,
                        default=1.,
                        help='resolution of endpoint choice in secs (default: 1')

    parser.add_argument('-i', '--gen-images', action='store_true',
                        help='produce image of each likelihood map (default: false)')

    parser.add_argument('-m', '--gen-maps', action='store_true',
                        help='produce map file for each likelihood map (default: false)')

    parser.add_argument('-n', '--nside', type=int,
                        default=64,
                        help='HEALPix NSide of output map (default: 64)')

    parser.add_argument('-o', '--output', type=Path,
                        default=Path('.'),
                        help='output path for maps/images (default: cwd)')

    parser.add_argument('-r', '--randseed', type=int,
                        default=1957,
                        help='random seed (default=1957)')

    parser.add_argument('-s', '--samples', default=None,
                        help='number of transients to sample from input path (default: all)')

    parser.add_argument('-t', '--nthreads', default=8,
                        help='number of CPU threads to use in mapping (default: 8)')

    parser.add_argument('-w', '--warmup', type=int,
                        default=3,
                        help='# warmup iterations before mapping (default: 3)')

    parser.add_argument('--bkg-model', type=Path,
                        default=default_bkg_model_path,
                        help=f'path to background file (default: {default_bkg_model_path})')

    parser.add_argument('--response', type=Path,
                        default=default_response_path,
                        help=f'path to response file (default: {default_response_path})')

    parser.add_argument('--training-times', type=Path,
                        default=default_mapping_times_path,
                        help=f'path to training data for mapping times (default: {default_mapping_times_path})')

    parser.add_argument('--training-outcomes', type=Path,
                        default=default_success_probs_path,
                        help=f'path to training data for mapping outcomes (default: {default_success_probs_path})')

    return vars(parser.parse_args(argv[1:]))
