[![Docker Publish](https://github.com/Gersbachlab-Bioinformatics/CLEANSER/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Gersbachlab-Bioinformatics/CLEANSER/actions/workflows/docker-publish.yml)

# CLEANSER

**C**rispr **L**ibrary **E**valuation and **A**mbient **N**oise **S**uppression for **E**nhanced sc**R**NA-seq
CLEANSER is a gRNA-cell assignment method that uses a mixture of two distinct distributions to model ambient and native gRNA presence in perturb-seq CRISPR libraries. CLEANSER takes into account gRNA-specific and cell-specific biases and generates a probability value of whether or not a gRNA is expressed natively in a cell.

## Installation

**Note**: These steps have been tested on macOS and Linux, but not Windows.

    pip install git+https://github.com/Gersbachlab-Bioinformatics/CLEANSER.git
    install_cmdstan

CLEANSER depends on something called [CmdStan](https://mc-stan.org/docs/cmdstan-guide/index.html) which, if you don't have, you'll need to install. Fortunately a script was installed as part of CLEANSER to make this easier. To install CmdStan run `install_cmdstan` which will download and install the latest version of CmdStan for CLEANSER to use.

## Usage

    cleanser [-h] -i INPUT [-o POSTERIORS_OUTPUT] [--so SO] [-n NUM_SAMPLES] [-w NUM_WARMUP] [-s SEED] [-c CHAINS]
                        [-p PARALLEL_RUNS] [--lpf NORMALIZATION_LPF] (--dc | --cs)

`-h`, `--help`: show the help message and exit

`-i INPUT`, `--input INPUT`: Matrix Market file of [guide library information](#input-file-format)

`-o POSTERIORS_OUTPUT`, `--posteriors-output POSTERIORS_OUTPUT`: output file name of per-guide/cell posterior probabilities

`--so SO`, `--samples-output SO`: output file name of sample data

`-n NUM_SAMPLES`, `--num-samples NUM_SAMPLES`: The number of samples to take of the model.

`-w NUM_WARMUP`, `--num-warmup NUM_WARMUP`: The number of warmup iterations per chain. Used by STAN for [automatic parameter tuning](https://mc-stan.org/docs/reference-manual/hmc-algorithm-parameters.html#automatic-parameter-tuning)

`-s SEED`, `--seed SEED`: The seed for the random number generator (This parameter will be used by STAN).

`-c CHAINS`, `--chains CHAINS`: The [number of Markov chains](https://mc-stan.org/docs/cmdstan-guide/mcmc-intro.html#multi-chain-sampling) (This parameter will be used by STAN).

`-p PARALLEL_RUNS`, `--parallel-runs PARALLEL_RUNS`: Number of guide models to run in parallel (this parameter will be used by STAN)/

`--lpf NORMALIZATION_LPF`, `--normalization-lpf NORMALIZATION_LPF`: The upper limit for including the guide counts in guide count normalization. Set to 0 for no limit. (LPF stands for "low pass filter")

`--dc`, `--direct-capture`: Use mixture model for direct capture experiments. Must specify either this or `--crop-seq`

`--cs`, `--crop-seq`: Use mixture model for crop-seq experiments. Must specify either this or `--direct-capture`.


### Using CLEANSER with Cell Ranger

The output from Cell Ranger can't be used directly by CLEANSER. The matrix market file that Cell Ranger
outputs includes entries for "Gene Expression" values and we don't want those in the CLEANSER input; we
only want the "CRISPR Guide Capture" entries. To create a new matrix market file with only the "CRISPR
Guide Capture" values use the `cr2cleanser` utility included with CLEANSER.

    cr2cleanser [-h] -m MATRIX_MARKET -f FEATURES [-o OUTPUT]

    Generate a MM file suitable for CLEANSER from Cell Ranger outputs

`-h`, `--help`: show this help message and exit
`-m MATRIX_MARKET`, `--matrix-market MATRIX_MARKET`: Cell Ranger matrix market output file
`-f FEATURES`, `--features FEATURES`: Cell Ranger features output
`-o OUTPUT`, `--output OUTPUT`: output file for use by CLEANSER

## Input File Format

The input file has a [Matrix Market file](https://math.nist.gov/MatrixMarket/formats.html#MMformat)-esque format where the column values are `Guide ID`, `Cell ID`, and `Guide Count`, in that order.

## Quality Control

A command, `cleanser_qc`, that will run a script to output QC information has been included.

```
usage: cleanser_qc [-h] -i INPUT -o OUTPUT_DIRECTORY [-g GUIDE_COUNTS] [-s SAMPLES] [-t THRESHOLD]

Generate CLEANSER QC information

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Cleanser posterior output file
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        Cleanser QC output directory
  -g GUIDE_COUNTS, --guide-counts GUIDE_COUNTS
                        Guide count file. Needed for UMI histogram and scatterplots
  -s SAMPLES, --samples SAMPLES
                        Cleanser sampling data. Needed for sample mean, variance, and mean histogram
  -t THRESHOLD, --threshold THRESHOLD
                        Disregard assignment probabilities below this value
```

`--input` and `--output-directory` are required, everything else is optional. When run including all the possible input files it will generate the following information:

- MOI
- Coverage
- Sample means
- A histogram of sample means
- Sample variance
- A scatterplot of UMI/posteriors
- A scatterplot of UMI/posteriors with UMIs on a log2 scale
- UMI histogram
- eCDF plot of posteriors

## Citation
If you use CLEANSER in your reseach, please cite:
> Liu S, Hamilton MC, Cowart T, Barrera A, Bounds LR, Nelson AC, Dornbaum SF, Riley JW, Doty RW, Allen AS, Crawford GE, Majoros WH, Gersbach CA. Characterization and bioinformatic filtering of ambient gRNAs in single-cell CRISPR screens using CLEANSER. Cell Genom. 2025 Feb 12;5(2):100766. doi: 10.1016/j.xgen.2025.100766. Epub 2025 Feb 5. PMID: 39914388; PMCID: PMC11872138.

The `main` branch should be preferentially used, but if you want to use the closest version of the software that was used in the manuscript above, please use the [v0.1](https://github.com/Gersbachlab-Bioinformatics/CLEANSER/tree/v0.1) branch.
