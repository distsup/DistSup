## Setting up the environment

1. Download and install [`miniconda 3`](https://docs.conda.io/en/latest/miniconda.html).
2. Add conda to `.bashrc` during installation, or run manually
   `eval "$(/home/jch/scratch/jsalt/anaconda3/bin/conda shell.bash hook)"`
   to populate your shell with conda programs.
3. Install our `conda` environment with `conda env create -f environment.yml`.
   To update the environment use `conda env update --file environment.yml`.

## Loading the environment

If the envirnonment has not been loaded in the current session, then:

1. Add the conda env `conda activate distsup`.
2. Add our env variables `source set-env.sh`.
