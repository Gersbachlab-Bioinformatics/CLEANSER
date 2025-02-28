# =========================================================================
# This is OPEN SOURCE SOFTWARE governed by the Gnu General Public
# License (GPL) version 3, as described at www.opensource.org.
# Copyright (C)2023 Siyan Liu (siyan.liu432@duke.edu)
# =========================================================================

import concurrent.futures
from collections import defaultdict
from importlib.resources import files
from operator import itemgetter

from cmdstanpy import CmdStanModel

from .configuration import Configuration, MMLine
from .constants import (
    DEFAULT_CHAINS,
    DEFAULT_NORM_LPF,
    DEFAULT_RUNS,
    DEFAULT_SAMPLE,
    DEFAULT_SEED,
    DEFAULT_WARMUP,
    MAX_SEED_INT,
)

CountData = dict[str, float]


def mm_counts(mtx_lines: list[MMLine], norm_lpf: int) -> tuple[dict[str, int], dict[str, list[tuple[str, int]]]]:
    cumulative_counts = {}
    per_guide_counts = defaultdict(lambda: [])

    for guide, cell_id, guide_count in mtx_lines:
        if norm_lpf:
            if cell_id not in cumulative_counts:
                cumulative_counts[cell_id] = 0

            if guide_count <= norm_lpf:
                cumulative_counts[cell_id] += guide_count
        else:
            if cell_id not in cumulative_counts:
                cumulative_counts[cell_id] = guide_count
            else:
                cumulative_counts[cell_id] += guide_count

        per_guide_counts[guide].append((cell_id, guide_count))

    for key, value in cumulative_counts.items():
        if value == 0:
            cumulative_counts[key] = 1

    return cumulative_counts, per_guide_counts


def normalize(count_data: dict[str, int]) -> CountData:
    count = len(count_data)
    total_size = sum(count_data.values())
    avg_size = total_size / count

    norm_cell_counts = {cell_id: lib_size / avg_size for cell_id, lib_size in count_data.items()}

    return norm_cell_counts


def run_stan(stan_args):
    model, guide_id, X, L, num_warmup, num_samples, chains, seed = stan_args
    fit = model.sample(
        data={"N": len(X), "X": X, "L": L},
        iter_warmup=num_warmup,
        iter_sampling=num_samples,
        chains=chains,
        seed=seed,
        show_progress=False,
    )
    return guide_id, fit


def run(
    config: Configuration,
    chains: int = DEFAULT_CHAINS,
    normalization_lpf: int = DEFAULT_NORM_LPF,
    num_parallel_runs: int = DEFAULT_RUNS,
    num_samples: int = DEFAULT_SAMPLE,
    num_warmup: int = DEFAULT_WARMUP,
    seed: int = DEFAULT_SEED,
):
    sorted_mm_lines = sorted(list(config.gen_data()), key=itemgetter(0, 1))
    cumulative_counts, per_guide_counts = mm_counts(sorted_mm_lines, normalization_lpf)
    normalized_counts = normalize(cumulative_counts)

    def stan_params():
        for guide_id, guide_counts in per_guide_counts.items():
            result = (
                CmdStanModel(stan_file=files("cleanser").joinpath(config.model)),
                guide_id,
                [guide_count for _, guide_count in guide_counts],  # X
                [normalized_counts[cell_id] for cell_id, _ in guide_counts],  # L
                num_warmup,
                num_samples,
                chains,
                (seed + int(guide_id)) % MAX_SEED_INT,
            )
            yield result

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_runs) as executor:
        for guide_id, samples in executor.map(run_stan, stan_params()):
            config.collect_samples(guide_id, samples)
            config.collect_stats(samples)
            config.output_posteriors(guide_id, samples, per_guide_counts[guide_id])
