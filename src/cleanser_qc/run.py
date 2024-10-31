import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from math import log2
from pathlib import Path
from statistics import mean, variance
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cleanser.run import MMLines, read_mm_file


@dataclass
class DCSampleMetadata:
    "Direct capture sample information"

    guide_id: str
    r: float
    mu: float
    dispersion: float
    neg_binom_mean: float
    neg_binom_dispersion: float


@dataclass
class CSSampleMetadata:
    "CROP-seq sample information"

    guide_id: str
    r: float
    mu: float
    dispersion: float
    lamb: float  # lambda


@dataclass
class Prediction:
    "CLEANSER Prediction for a guide/cell pair"
    guide_id: str
    cell_id: str
    prediction: float


def per_guide_sample_stats(
    samples: list[CSSampleMetadata] | list[DCSampleMetadata],
) -> tuple[list[CSSampleMetadata] | list[DCSampleMetadata], list[CSSampleMetadata] | list[DCSampleMetadata]]:

    sample_count = len(samples)

    if sample_count == 0:
        raise ValueError("No samples")

    if isinstance(samples[0], CSSampleMetadata):
        sample_tallies = defaultdict(lambda: [[], [], [], []])
        for sample in cast(list[CSSampleMetadata], samples):
            running_tally = sample_tallies[sample.guide_id]
            running_tally[0].append(sample.r)
            running_tally[1].append(sample.mu)
            running_tally[2].append(sample.dispersion)
            running_tally[3].append(sample.lamb)

        means = [
            CSSampleMetadata(guide_id, mean(tally[0]), mean(tally[1]), mean(tally[2]), mean(tally[3]))
            for guide_id, tally in sample_tallies.items()
        ]

        variances = [
            CSSampleMetadata(guide_id, variance(tally[0]), variance(tally[1]), variance(tally[2]), variance(tally[3]))
            for guide_id, tally in sample_tallies.items()
        ]

        return means, variances

    if isinstance(samples[0], DCSampleMetadata):
        sample_tallies = defaultdict(lambda: [[], [], [], [], []])
        for sample in cast(list[DCSampleMetadata], samples):
            running_tally = sample_tallies[sample.guide_id]
            running_tally[0].append(sample.r)
            running_tally[1].append(sample.mu)
            running_tally[2].append(sample.dispersion)
            running_tally[3].append(sample.neg_binom_mean)
            running_tally[4].append(sample.neg_binom_dispersion)

        means = [
            DCSampleMetadata(
                guide_id,
                mean(tally[0]),
                mean(tally[1]),
                mean(tally[2]),
                mean(tally[3]),
                mean(tally[4]),
            )
            for guide_id, tally in sample_tallies.items()
        ]

        variances = [
            DCSampleMetadata(
                guide_id,
                variance(tally[0]),
                variance(tally[1]),
                variance(tally[2]),
                variance(tally[3]),
                variance(tally[4]),
            )
            for guide_id, tally in sample_tallies.items()
        ]

        return means, variances

    raise ValueError("Invalid sample type")


def plot_hist(values, bin_count, title="", x_label="", y_label="", density=False) -> Figure:
    fig, ax = plt.subplots()

    # the histogram of the data
    ax.hist(values, bins=bin_count, density=density)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    return fig


def plot_scatter(x, y, title="", x_label="", y_label="") -> Figure:
    fig, ax = plt.subplots()

    # the histogram of the data
    ax.scatter(x, y, s=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    return fig


def plot_ecdf(x, threshold: float, title="", x_label="", y_label="") -> Figure:
    # plot:
    fig, ax = plt.subplots()

    ax.ecdf(x)

    if threshold != 0.0:
        ax.plot((threshold, threshold), (0, 1), linestyle="dashed", label="Threshold")
        ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    return fig


def sample_stat_output(statistics: list[CSSampleMetadata] | list[DCSampleMetadata]) -> str:
    statistic_text = StringIO()

    if isinstance(statistics[0], CSSampleMetadata):
        statistic_text.write("guide\tr\tmu\tdisp\tlambda\n")
        for ave in cast(list[CSSampleMetadata], statistics):
            statistic_text.write(f"{ave.guide_id}\t{ave.r}\t{ave.mu}\t{ave.dispersion}\t{ave.lamb}\n")

    if isinstance(statistics[0], DCSampleMetadata):
        statistic_text.write("guide\tr\tmu\tdisp\tn_mu\tn_disp\n")
        for ave in cast(list[DCSampleMetadata], statistics):
            statistic_text.write(
                f"{ave.guide_id}\t{ave.r}\t{ave.mu}\t{ave.dispersion}\t{ave.neg_binom_mean}\t{ave.neg_binom_dispersion}\n"
            )

    return statistic_text.getvalue()


def sample_mean_histogram(means: list[CSSampleMetadata] | list[DCSampleMetadata]) -> Figure:
    mean_count = len(means)

    if mean_count == 0:
        raise ValueError("No means")

    mean_r = [a.r for a in means]
    fig = plot_hist(mean_r, 30, title="cleanser", x_label="sample mean", y_label="count")
    return fig


def assigned_counts_histogram(predictions: list[Prediction], mm_lines: MMLines, threshold: float) -> Figure:
    thresholded_preds = {(p.guide_id, p.cell_id) for p in predictions if p.prediction >= threshold}
    umis = [umi for guide_id, cell_id, umi in mm_lines if (guide_id, cell_id) in thresholded_preds]
    fig = plot_hist(umis, 100, title="cleanser", x_label="UMI", y_label="count")
    return fig


def posterior_umi_scatterplot(predictions: list[Prediction], mm_lines: MMLines) -> Figure:
    umi_keys = [(p.guide_id, p.cell_id) for p in predictions]
    posteriors = [p.prediction for p in predictions]

    # order the umis to match the posteriors
    umi_dict = {(guide_id, cell_id): umi for guide_id, cell_id, umi in mm_lines}
    umis = [umi_dict[k] for k in umi_keys]

    fig = plot_scatter(posteriors, umis, title="cleanser", x_label="Posterior", y_label="UMI")

    return fig


def posterior_umi_scatterplot_log2(predictions: list[Prediction], mm_lines: MMLines) -> Figure:
    umi_keys = [(p.guide_id, p.cell_id) for p in predictions]
    posteriors = [p.prediction for p in predictions]

    # order the umis to match the posteriors
    umi_dict = {(guide_id, cell_id): log2(umi) for guide_id, cell_id, umi in mm_lines}
    umis = [umi_dict[k] for k in umi_keys]

    fig = plot_scatter(posteriors, umis, title="cleanser", x_label="Posterior", y_label="log2(UMI)")

    return fig


def prob_ecdf(predictions: list[Prediction], threshold: float) -> Figure:
    fig = plot_ecdf(
        [p.prediction for p in predictions],
        threshold=threshold,
        title="Cleanser",
        x_label="Probability of assignment",
        y_label="Count",
    )
    return fig


def calc_moi(predictions: list[Prediction], threshold: float) -> float:
    cells = set()
    predict_count = 0
    for prediction in predictions:
        cells.add(prediction.cell_id)
        if prediction.prediction >= threshold:
            predict_count += 1

    return predict_count / len(cells)


def calc_coverage(predictions: list[Prediction], threshold: float) -> float:
    guides = set()
    predict_count = 0
    for prediction in predictions:
        guides.add(prediction.guide_id)
        if prediction.prediction >= threshold:
            predict_count += 1

    return predict_count / len(guides)


def read_predictions(input_file) -> list[Prediction]:
    reader = csv.reader(input_file, delimiter="\t")

    return [Prediction(guide_id=l[0], cell_id=l[1], prediction=float(l[2])) for l in reader]


def read_sample_data(sample_data_file) -> list[CSSampleMetadata] | list[DCSampleMetadata]:
    reader = csv.DictReader(sample_data_file, delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError("Invalid sample data file")

    if len(reader.fieldnames) == 5:
        return [
            CSSampleMetadata(l["guide id"], float(l["r"]), float(l["mu"]), float(l["Disp"]), float(l["lambda"]))
            for l in reader
        ]

    if len(reader.fieldnames) == 6:
        return [
            DCSampleMetadata(
                l["guide id"],
                float(l["r"]),
                float(l["mu"]),
                float(l["Disp"]),
                float(l["n_nbMean"]),
                float(l["n_nbDisp"]),
            )
            for l in reader
        ]

    raise ValueError("Invalid sample data file")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLEANSER QC information")
    parser.add_argument("-i", "--input", type=argparse.FileType(), help="Cleanser posterior output file", required=True)
    parser.add_argument("-o", "--output-directory", help="Cleanser QC output directory", required=True)
    parser.add_argument(
        "-g",
        "--guide-counts",
        type=argparse.FileType(),
        help="Guide count file. Needed for UMI histogram and scatterplots",
    )
    parser.add_argument(
        "-s",
        "--samples",
        type=argparse.FileType(),
        help="Cleanser sampling data. Needed for sample mean, variance, and mean histogram",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.0, help="Disregard assignment probabilities below this value"
    )

    return parser.parse_args()


def write(output_dir, filename, contents):
    output_file = output_dir / Path(filename)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(contents)


def run_cli():
    args = get_args()

    output_dir = Path(args.output_directory)
    if not output_dir.exists():
        output_dir.mkdir()
    elif not output_dir.is_dir():
        raise ValueError(f"Output directory {output_dir} must be a directory")

    predictions = read_predictions(args.input)
    moi = calc_moi(predictions, args.threshold)
    moi_text = f"MOI: {moi}"
    print(moi_text)
    write(output_dir, "moi.txt", moi_text + "\n")

    coverage = calc_coverage(predictions, args.threshold)
    coverage_text = f"Coverage: {coverage}"
    print(coverage_text)
    write(output_dir, "coverage.txt", coverage_text + "\n")

    prob_ecdf(predictions, args.threshold)
    plt.savefig(output_dir / Path("ecdf.png"))

    if args.guide_counts is not None:
        mm_lines = read_mm_file(args.guide_counts)
        assigned_counts_histogram(predictions, mm_lines, args.threshold)
        plt.savefig(output_dir / Path("umi_hist.png"))

        posterior_umi_scatterplot(predictions, mm_lines)
        plt.savefig(output_dir / Path("umi_count_scatter.png"))

        posterior_umi_scatterplot_log2(predictions, mm_lines)
        plt.savefig(output_dir / Path("umi_count_scatter_log2.png"))

    if args.samples is not None:
        samples = read_sample_data(args.samples)
        assert len(samples) > 0
        means, variances = per_guide_sample_stats(samples)
        write(output_dir, "sample_mean.txt", sample_stat_output(means))
        write(output_dir, "sample_variance.txt", sample_stat_output(variances))
        sample_mean_histogram(means)
        plt.savefig(output_dir / Path("sample_mean_hist.png"))
