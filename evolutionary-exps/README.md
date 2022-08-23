# Evolutionary proxy experiments

The `evolutionary-exps` subdirectory contains the code used to run evolutionary proxy experiments presented
in the paper.

## Prerequisites

- Mention Python version
- Mention that Node is required
- Mention Prudens, automatically downloaded
- Has been tested on Linux, but will probably run on macOS/Windows without issue
- Warning about training times
- multiprocessing, number of cores

## Installing dependencies

Project dependencies are specified in `pyproject.toml`, and can be installed using [Poetry](https://python-poetry.org/)
(installation instructions [here](https://python-poetry.org/docs/#installation)), using:

```bash
cd evolutionary-exps
poetry install
```

## Running the experiments

After dependencies have been installed, activate the generated virtual environment using:

```bash
poetry shell
```

create python script in evolutionary-exps that will contain the cli interface

custom data dir

custom number of processes

add description of the cli interface (--help etc.)

### Basic

how to run experiments from paper

```bash
python evol_proxy.py
```

### Custom

custom data, follow the instructions here (link to instructions from js part)

```bash
python evol_proxy.py -d "absolute/path/" -o "results/path"
```

### Results

Results will be written at ...
