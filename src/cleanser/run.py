import argparse
import sys

from .configuration import Model, MtxConfiguration, MuDataConfiguration
from .constants import (
    DEFAULT_CHAINS,
    DEFAULT_NORM_LPF,
    DEFAULT_RUNS,
    DEFAULT_SAMPLE,
    DEFAULT_SEED,
    DEFAULT_WARMUP,
)
from .guide_mixture import run


def get_args():
    parser = argparse.ArgumentParser(
        "cleanser",
        description="Crispr Library Evaluation and Ambient Noise Suppression for Enhanced scRNA-seq",
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Matrix Market or MuData file of guide library information",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--posteriors-output",
        help="output file name of per-guide/cell posterior probabilities. Required for MuData inputs (will be another MuData file)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--so",
        "--samples-output",
        help="output file name of sample data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=DEFAULT_SAMPLE, help="The number of samples to take of the model"
    )
    parser.add_argument(
        "-w", "--num-warmup", type=int, default=DEFAULT_WARMUP, help="The number of warmup iterations per chain"
    )
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help="The seed for the random number generator")
    parser.add_argument("-c", "--chains", type=int, default=DEFAULT_CHAINS, help="The number of Markov chains")
    parser.add_argument(
        "-p",
        "--parallel-runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of guide models to run in parallel",
    )
    parser.add_argument(
        "--lpf",
        "--normalization-lpf",
        type=int,
        default=DEFAULT_NORM_LPF,
        help="The upper limit for including the guide counts in guide count normalization. Set to 0 for no limit.",
        dest="normalization_lpf",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--dc",
        "--direct-capture",
        action="store_true",
        help="Use direct capture mixture model",
    )
    model_group.add_argument(
        "--cs",
        "--crop-seq",
        action="store_true",
        help="Use crop-seq mixture model",
    )
    mudata_group = parser.add_argument_group("MuData")
    mudata_group.add_argument("--modality", help="The name of the MuData modality the guide information is in")
    mudata_group.add_argument(
        "--output-layer",
        help="The name of the layer (under the same modality as the input) to put the posterior probability data in",
    )
    mudata_group.add_argument(
        "--capture-method-key",
        help="The key for accessing the capture method name from the modalities unstructured data",
    )
    parser.add_argument(
        "-t", "--threshold", help="If set, the guide calls will be binarized at this cutoff", default=None, type=float
    )

    return parser.parse_args()


def get_configuration(args):
    input_filename = args.input
    match input_filename.split(".")[-1]:
        case "mm" | "mtx":
            if args.dc:
                model = Model.DC
            elif args.cs:
                model = Model.CS
            else:
                raise argparse.ArgumentError(
                    argument=None, message="Exactly one of --direct-capture or --crop-seq arguments is required."
                )

            return MtxConfiguration(
                input=args.input, model=model, sample_output=args.so, posteriors_output=args.posteriors_output
            )  # matrix market
        case "h5mu" | "h5ad" | "h5" | "hdf5" | "he5":
            if args.capture_method_key is None:
                if args.dc:
                    model = Model.DC
                elif args.cs:
                    model = Model.CS
                else:
                    raise argparse.ArgumentError(
                        argument=None,
                        message="Must specify either a capture method (--direct-capture or --crop-seq arguments) or the key to get the capture method from the mudata file.",
                    )

            else:
                model = None

            if args.modality is None:
                raise argparse.ArgumentError(
                    argument=None, message="The --modality argument is required for MuData files."
                )

            if args.output_layer is None:
                raise argparse.ArgumentError(
                    argument=None, message="The --output-layer argument is required for MuData files."
                )

            if args.posteriors_output is None:
                raise argparse.ArgumentError(
                    argument=None, message="The --posteriors-ouput argument is required for MuData files."
                )

            return MuDataConfiguration(
                input=args.input,
                modality=args.modality,
                capture_method=args.capture_method_key,
                output_layer=args.output_layer,
                model=model,
                sample_output=args.so,
                posteriors_output=args.posteriors_output,
                threshold=args.threshold,
            )
    raise ValueError("Invalid input file type. Please input uncompressed Matrix Market or MuData files only.")


def run_cli():
    args = get_args()
    configuration = get_configuration(args)

    try:
        run(
            configuration,
            chains=args.chains,
            normalization_lpf=args.normalization_lpf,
            num_parallel_runs=args.parallel_runs,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            seed=args.seed,
        )

        configuration.output_samples()
        configuration.output_posteriors()
        configuration.output_stats()

        print(f"Random seed: {args.seed}")
    except KeyboardInterrupt:
        sys.exit(1)
