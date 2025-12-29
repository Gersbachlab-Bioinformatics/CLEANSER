import multiprocessing as mp
from random import randint

CS_MODEL_FILE = "cs-guide-mixture.stan"
DC_MODEL_FILE = "dc-guide-mixture.stan"
MAX_SEED_INT = 4_294_967_295  # 2^32 - 1, the largest seed allowed by STAN

DEFAULT_CHAINS = 4
DEFAULT_NORM_LPF = 2
DEFAULT_RUNS = mp.cpu_count()
DEFAULT_SAMPLE = 1000
DEFAULT_SEED = randint(0, MAX_SEED_INT)
DEFAULT_WARMUP = 300

MUDATA_POSTERIORS_LAYER_SUFFIX = "posteriors"
