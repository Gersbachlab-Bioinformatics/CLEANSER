from collections.abc import Generator
from os import SEEK_SET

import mudata as md
from scipy.sparse import dok_matrix


__all__ = ["MtxConfiguration", "MuDataConfiguration"]

MMLine = tuple[str, str, int]
Posterior = tuple[str, str, float]
PData = Generator[Posterior, None, None]
MMData = Generator[MMLine, None, None]


class Configuration:
    def gen_count_data(self) -> MMData:
        raise NotImplementedError("This is an abstract method")

    def gen_posteriors_data(self) -> PData:
        raise NotImplementedError("This is an abstract method")


class MuDataConfiguration(Configuration):
    guides = None

    def __init__(self, posterior_input, count_input, modality, posteriors_layer):
        if posterior_input is not None:
            self.posterior_input_file = md.read(posterior_input)
            p_guides = self.posterior_input_file[modality]
            self.posteriors = p_guides.layers[posteriors_layer]

        if count_input is not None:
            self.count_input_file = md.read(count_input)
            self.guides = self.count_input_file[modality]

    def gen_count_data(self) -> MMData:
        if self.guides is None:
            return

        guide_count_array = self.guides.X.todok()
        for key, guide_count in guide_count_array.items():
            yield (key[1], key[0], int(guide_count))

    def gen_posteriors_data(self) -> PData:
        if self.posterior_input_file is None:
            return

        guide_posteriors_array = self.posteriors.todok()
        for key, posteriors in guide_posteriors_array.items():
            yield (key[1], key[0], float(posteriors))


class MtxConfiguration(Configuration):
    def __init__(self, posterior_input, count_input):
        if posterior_input is not None:
            self.posterior_input_file = open(posterior_input, "r", encoding="utf8")
        else:
            self.posterior_input_file = None

        if count_input is not None:
            self.count_input_file = open(count_input, "r", encoding="utf8")
        else:
            self.count_input_file = None

    def __del__(self):
        if self.posterior_input_file is not None:
            self.posterior_input_file.close()

        if self.count_input_file is not None:
            self.count_input_file.close()

    def gen_count_data(self) -> MMData:
        if self.count_input_file is None:
            return

        for line in self.count_input_file:
            # Skip market matrix header/comments
            if line.startswith("%"):
                continue

            # store the first non-comment line. We'll re-use it
            # in the output.
            self.mm_header = line
            break

        for line in self.count_input_file:
            guide, cell, count = line.strip().split()
            yield (guide, cell, int(count))

        self.count_input_file.seek(0, SEEK_SET)

    def gen_posteriors_data(self) -> PData:
        if self.posterior_input_file is None:
            return

        for line in self.posterior_input_file:
            # Skip market matrix header/comments
            if line.startswith("%"):
                continue

            # store the first non-comment line. We'll re-use it
            # in the output.
            self.mm_header = line
            break

        for line in self.posterior_input_file:
            guide, cell, count = line.strip().split()
            yield (guide, cell, float(count))

        self.posterior_input_file.seek(0, SEEK_SET)
