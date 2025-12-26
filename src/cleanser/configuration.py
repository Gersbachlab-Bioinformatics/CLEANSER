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

    def collect_posteriors(self, guide_id, samples, cell_info):
        raise NotImplementedError("This is an abstract method")

    def output_posteriors(self):
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
    def __init__(
        self, input, modality, capture_method, output_layer, model, sample_output, posteriors_output, threshold
    ):
        super().__init__(input, model, sample_output, posteriors_output)
        self.input_file = md.read(input)
        self.guides = self.input_file[modality]
        if model is None:
            analysis = self.guides.uns.get(capture_method)
            if analysis is not None:
                if analysis[0] == "CROP-seq":
                    self.model = Model.CS
                elif analysis[0] == "direct capture":
                    self.model = Model.DC
        else:
            self.model = model
        self.output_layer = output_layer
        self.output_matrix = dok_matrix(self.guides.X.shape)
        self.posteriors_output_file = posteriors_output
        self.threshold = threshold

        if threshold is None:
            self.collect_posteriors = self._raw_collect
        else:
            self.output_binary_matrix = dok_matrix(self.guides.X.shape)
            self.collect_posteriors = self._raw_and_threshold_collect
                            
    def __del__(self):
        sample_output_file = getattr(self, "sample_output_file", None)
        if sample_output_file is not None and getattr(sample_output_file, "close", None) is not None:
            sample_output_file.close()

    def gen_data(self) -> MMData:
        guide_count_array = self.guides.X.todok()
        for key, guide_count in guide_count_array.items():
            yield (key[1], key[0], int(guide_count))

    def _raw_and_threshold_collect(self, guide_id, samples, cell_info):
        pzi = np.transpose(samples.stan_variable("PZi"))
        for i, (cell_id, _) in enumerate(cell_info):
            self.output_matrix[cell_id, guide_id] = np.median(pzi[i])
            if np.median(pzi[i]) >= self.threshold:
                self.output_binary_matrix[cell_id, guide_id] = 1

    def _raw_collect(self, guide_id, samples, cell_info):
        pzi = np.transpose(samples.stan_variable("PZi"))
        for i, (cell_id, _) in enumerate(cell_info):
            self.output_matrix[cell_id, guide_id] = np.median(pzi[i])

    def _threshold_collect(self, guide_id, samples, cell_info):
        pzi = np.transpose(samples.stan_variable("PZi"))
        for i, (cell_id, _) in enumerate(cell_info):
            if np.median(pzi[i]) >= self.threshold:
                self.output_matrix[cell_id, guide_id] = 1

    def output_posteriors(self):
        if self.threshold is not None:
            self.guides.layers[self.output_layer] = self.output_binary_matrix.tocsr()
            self.guides.layers[f"{self.output_layer}_posteriors"] = self.output_matrix.tocsr()
        else:
            self.guides.layers[self.output_layer] = self.output_matrix.tocsr()
        md.write(self.posteriors_output_file, self.input_file)


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
        input_file = getattr(self, "input_file", None)
        if input_file is not None:
            input_file.close()
        sample_output_file = getattr(self, "sample_output_file", None)
        if sample_output_file is not None and getattr(sample_output_file, "close", None) is not None:
            sample_output_file.close()
        posteriors_output_file = getattr(self, "posteriors_output_file", None)
        if posteriors_output_file is not None and getattr(posteriors_output_file, "close", None) is not None:
            posteriors_output_file.close()

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

    def collect_posteriors(self, guide_id, samples, cell_info):
        if self.mm_header is not None:
            self.posteriors_output_file.write(self.mm_header)
            self.mm_header = None

        pzi = np.transpose(samples.stan_variable("PZi"))
        for i, (cell_id, _) in enumerate(cell_info):
            if self.output_all_posteriors:
                with open(f"posteriors/{guide_id}_{cell_id}.txt", "w", encoding="ascii") as post_out:
                    post_out.write(f"{', '.join(str(n) for n in sorted(pzi[i]))}")

            self.posteriors_output_file.write(f"{guide_id}\t{cell_id}\t{np.median(pzi[i])}\n")

    def output_posteriors(self):
        if not self.output_all_posteriors:
            print("Please create a 'posteriors' directory if you want all posterior values saved.")
