# =========================================================================
# This is OPEN SOURCE SOFTWARE governed by the Gnu General Public
# License (GPL) version 3, as described at www.opensource.org.
# Copyright (C)2023 Siyan Liu (siyan.liu432@duke.edu)
# =========================================================================

import concurrent.futures
import os
from collections import defaultdict
from collections.abc import Iterable
from importlib.resources import files

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


def mm_counts(mtx_lines: Iterable[MMLine], norm_lpf: int) -> tuple[dict[str, int], dict[str, list[tuple[str, int]]]]:
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


_worker_model = None


def _init_worker(model_file):
    # Constructing a CmdStanModel checks the compiled binary's hash/timestamp
    # against the .stan source, which is real (if modest) work. Doing this once
    # per worker process here -- instead of once per guide in the main process,
    # serially, ahead of the parallel section -- removes a redundant cost that
    # otherwise scales linearly with guide count for no benefit.
    global _worker_model
    _worker_model = CmdStanModel(stan_file=files("cleanser").joinpath(model_file))


def run_stan(stan_args):
    guide_id, X, L, num_warmup, num_samples, chains, seed = stan_args
    fit = _worker_model.sample(
        data={"N": len(X), "X": X, "L": L},
        iter_warmup=num_warmup,
        iter_sampling=num_samples,
        chains=chains,
        seed=seed,
        show_progress=False,
    )

    return guide_id, fit


def delete_temp_files(samples):
    # CmdStan will leave the temp files it generates around until the python process exists
    # (Using the tempfile module). Because we are reusing the same python processes in the process
    # pool the whole run these temp files will really pile up, using possibly hundreds of GB of
    # space.
    #
    # This method deletes the temp files manually so we don't have that problem.

    for file in samples.runset.csv_files:
        os.remove(file)


def run(
    config: Configuration,
    chains: int = DEFAULT_CHAINS,
    normalization_lpf: int = DEFAULT_NORM_LPF,
    num_parallel_runs: int = DEFAULT_RUNS,
    num_samples: int = DEFAULT_SAMPLE,
    num_warmup: int = DEFAULT_WARMUP,
    seed: int = DEFAULT_SEED,
):
    # mm_counts() only accumulates into dicts keyed by guide/cell -- it has no
    # dependency on input order, so sorting (and the full extra in-memory copy
    # that entails) here was pure overhead at guide/cell counts large enough
    # for it to matter.
    cumulative_counts, per_guide_counts = mm_counts(config.gen_data(), normalization_lpf)
    normalized_counts = normalize(cumulative_counts)

    def stan_params():
        for guide_id, guide_counts in per_guide_counts.items():
            result = (
                guide_id,
                [guide_count for _, guide_count in guide_counts],  # X
                [normalized_counts[cell_id] for cell_id, _ in guide_counts],  # L
                num_warmup,
                num_samples,
                chains,
                (seed + int(guide_id)) % MAX_SEED_INT,
            )
            yield result

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_parallel_runs, initializer=_init_worker, initargs=(config.model,)
    ) as executor:
        futures = [executor.submit(run_stan, args) for args in stan_params()]
        # as_completed (not map, which yields in submission order) so a guide
        # that's slow to fit can't block cleanup of every other guide that
        # finished after it -- each guide's temp files are deleted the moment
        # that guide is done, bounding simultaneous disk usage to roughly
        # num_parallel_runs guides' worth regardless of how variable
        # individual guides' fit times are.
        for future in concurrent.futures.as_completed(futures):
            guide_id, samples = future.result()
            config.collect_samples(guide_id, samples)
            config.collect_stats(guide_id, samples)
            config.collect_posteriors(guide_id, samples, per_guide_counts[guide_id])

            delete_temp_files(samples)
