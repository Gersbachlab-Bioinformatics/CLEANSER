import os.path
import sys
from collections.abc import Generator
from enum import StrEnum
from typing import Optional, TextIO

import numpy as np
import mudata as md
from scipy.sparse import dok_matrix

from .constants import (
    CS_MODEL_FILE,
    DC_MODEL_FILE,
)

__all__ = ["Model", "MtxConfiguration", "MuDataConfiguration"]

MMLine = tuple[str, str, int]
MMData = Generator[MMLine, None, None]


class Model(StrEnum):
    CS = CS_MODEL_FILE
    DC = DC_MODEL_FILE


class Configuration:
    model: Model
    sample_output_file: TextIO
    samples: list[tuple]
    stats: list[tuple]

    def __init__(self, input, model, sample_output, posteriors_output):
        self.model = model
        self.samples = []
        self.stats = []

        if sample_output is not None:
            self.sample_output_file = open(sample_output, "w", encoding="utf8")
        else:
            self.sample_output_file = sys.stdout

    def gen_data(self) -> MMData:
        raise NotImplementedError("This is an abstract method")

    def output_sample(self, guide_id, samples):
        raise NotImplementedError("This is an abstract method")

    def output_posteriors(self, guide_id, samples, cell_info):
        raise NotImplementedError("This is an abstract method")

    def collect_samples(self, guide_id, samples):
        match self.model:
            case Model.DC:
                self.collect_dc_samples(guide_id, samples)
            case Model.CS:
                self.collect_cs_samples(guide_id, samples)

    def collect_cs_samples(self, guide_id, samples):
        r = samples.stan_variable("r")
        mu = samples.stan_variable("nbMean")
        disp = samples.stan_variable("nbDisp")
        lamb = samples.stan_variable("lambda")
        for i, r_samp in enumerate(r):
            self.samples.append((guide_id, r_samp, mu[i], disp[i], lamb[i]))

    def collect_dc_samples(self, guide_id, samples):
        r = samples.stan_variable("r")
        mu = samples.stan_variable("nbMean")
        disp = samples.stan_variable("nbDisp")
        n_mean = samples.stan_variable("n_nbMean")
        n_disp = samples.stan_variable("n_nbDisp")
        for i, r_samp in enumerate(r):
            self.samples.append((guide_id, r_samp, mu[i], disp[i], n_mean[i], n_disp[i]))

    def output_samples(self):
        match self.model:
            case Model.DC:
                self.output_dc_samples()
            case Model.CS:
                self.output_cs_samples()

    def output_cs_samples(self):
        self.sample_output_file.write("guide id\tr\tmu\tDisp\tlambda\n")
        for guide_id, r_samp, mu, disp, lam in self.samples:
            self.sample_output_file.write(f"{guide_id}\t{r_samp}\t{mu}\t{disp}\t{lam}\n")

    def output_dc_samples(self):
        self.sample_output_file.write("guide id\tr\tmu\tDisp\tn_nbMean\tn_nbDisp\n")
        for guide_id, r_samp, mu, disp, n_mean, n_disp in self.samples:
            self.sample_output_file.write(f"{guide_id}\t{r_samp}\t{mu}\t{disp}\t{n_mean}\t{n_disp}\n")

    def collect_stats(self, results):
        match self.model:
            case Model.DC:
                self.collect_dc_stats(results)
            case Model.CS:
                self.collect_cs_stats(results)

    def collect_cs_stats(self, samples):
        self.stats.append(
            (
                np.median(samples.stan_variable("r")),
                np.median(samples.stan_variable("nbMean")),
                np.median(samples.stan_variable("lambda")),
            )
        )

    def collect_dc_stats(self, samples):
        self.stats.append(
            (
                np.median(samples.stan_variable("r")),
                np.median(samples.stan_variable("nbMean")),
                np.median(samples.stan_variable("n_nbMean")),
                np.median(samples.stan_variable("n_nbDisp")),
            )
        )

    def output_stats(self):
        match self.model:
            case Model.DC:
                self.output_dc_stats()
            case Model.CS:
                self.output_cs_stats()

    def output_cs_stats(self):
        for r, mu, lam in self.stats:
            print(f"r={r}\tmu={mu}\tlambda={lam}")

    def output_dc_stats(self):
        for r, mu, n_nbMean, n_nbDisp in self.stats:
            print(f"r={r}\tmu={mu}\n_nbMean={n_nbMean}\tn_nbDisp={n_nbDisp}")


class MuDataConfiguration(Configuration):
    def __init__(self, input, model, sample_output, posteriors_output):
        self.file = md.read(input)
        if args.dc:
            self.model = Model.DC
        elif args.cs:
            self.model = Model.CS
        self.sample_output_file = open(sample_output, "w", encoding="utf8")
        self.posteriors_output_file = posteriors_output
        # md.write(args.output, mu_input)

    def posteriors_layer(self, stan_results, array, threshold=None):
        if threshold is None:
            for guide_id, (samples, cell_info) in stan_results.items():
                pzi = np.transpose(samples.stan_variable("PZi"))
                for i, (cell_id, _) in enumerate(cell_info):
                    array[cell_id, guide_id] = np.median(pzi[i])
        else:
            for guide_id, (samples, cell_info) in stan_results.items():
                pzi = np.transpose(samples.stan_variable("PZi"))
                for i, (cell_id, _) in enumerate(cell_info):
                    if np.median(pzi[i]) >= threshold:
                        array[cell_id, guide_id] = 1

        return array.tocsr()

    def cleanser_posteriors(self, guides, threshold):
        guide_count_array = guides.X.todok()
        counts = [(key[1], key[0], int(guide_count)) for key, guide_count in guide_count_array.items()]
        analysis = guides.uns.get("capture_method")
        if analysis is None or analysis[0] == "CROP-seq":
            model = CS_MODEL_FILE
        elif analysis == "direct capture":
            model = DC_MODEL_FILE
        else:
            raise ValueError("Invalid capture method type")

        results = asyncio.run(run_cleanser(counts, model))
        return posteriors_layer(results, dok_matrix(guides.X.shape), threshold)


class MtxConfiguration(Configuration):
    mm_header: Optional[str] = None

    def __init__(self, input, model, sample_output, posteriors_output):
        super().__init__(input, model, sample_output, posteriors_output)
        self.input_file = open(input, "r", encoding="utf8")

        if posteriors_output is not None:
            self.posteriors_output_file = open(posteriors_output, "w", encoding="utf8")
        else:
            self.posteriors_output_file = sys.stdout

        self.output_all_posteriors = os.path.isdir("posteriors")

    def __del__(self):
        if not self.output_all_posteriors:
            print("Please create a 'posteriors' directory if you want all posterior values saved.")

        self.input_file.close()
        if self.sample_output_file != sys.stdout:
            self.sample_output_file.close()
        if self.posteriors_output_file != sys.stdout:
            self.posteriors_output_file.close()

    def gen_data(self) -> MMData:
        for line in self.input_file:
            # Skip market matrix header/comments
            if line.startswith("%"):
                continue

            # store the first non-comment line. We'll re-use it
            # in the output.
            self.mm_header = line
            break

        for line in self.input_file:
            guide, cell, count = line.strip().split()
            yield (guide, cell, int(count))

    def output_posteriors(self, guide_id, samples, cell_info):
        if self.mm_header is not None:
            self.posteriors_output_file.write(self.mm_header)
            self.mm_header = None

        pzi = np.transpose(samples.stan_variable("PZi"))
        for i, (cell_id, _) in enumerate(cell_info):
            if self.output_all_posteriors:
                with open(f"posteriors/{guide_id}_{cell_id}.txt", "w", encoding="ascii") as post_out:
                    post_out.write(f"{', '.join(str(n) for n in sorted(pzi[i]))}")

            self.posteriors_output_file.write(f"{guide_id}\t{cell_id}\t{np.median(pzi[i])}\n")
