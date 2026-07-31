import multiprocessing as mp
import os
from random import randint

CS_MODEL_FILE = "cs-guide-mixture.stan"
DC_MODEL_FILE = "dc-guide-mixture.stan"
MAX_SEED_INT = 4_294_967_295  # 2^32 - 1, the largest seed allowed by STAN


def _available_cpu_count() -> int:
    # mp.cpu_count() (like os.cpu_count()) reports the PHYSICAL node's total
    # CPU count, not what a job scheduler's cgroup actually allocated to this
    # process. On a shared HPC node this is a real trap: a job that requests
    # e.g. 1 core on a 64-core node would still default to spawning ~64
    # parallel worker processes (each launching its own multi-chain CmdStan
    # subprocess group), massively oversubscribing the cgroup's actual CPU
    # entitlement. This showed up in practice as extreme memory usage and
    # severely disproportionate wall-clock time on a real production run.
    # os.sched_getaffinity(0) (Linux only) correctly reports the cgroup/cpuset
    # restricted count instead; fall back to mp.cpu_count() where it doesn't
    # exist (e.g. macOS, Windows).
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


DEFAULT_CHAINS = 4
DEFAULT_NORM_LPF = 2
DEFAULT_RUNS = _available_cpu_count()
DEFAULT_SAMPLE = 1000
DEFAULT_SEED = randint(0, MAX_SEED_INT)
DEFAULT_WARMUP = 300

MUDATA_POSTERIORS_LAYER_SUFFIX = "posteriors"
