# Evolutionary proxy coach experiments

This subdirectory contains the code used to run evolutionary proxy experiments presented
in the paper. The instructions below outline how to replicate the experiments, or run new ones.

## Prerequisites

- Python >= 3.10 (versions >= 3.6 may also work, but have not been tested).
- [Node.js](https://nodejs.org/en/download/) must be installed and available on PATH.
- Tested to work on Linux (macOS/Windows can probably be used without issue, but have not been tested).
- Individual experiments may take 5 to 60+ min to run (depending on parameters such as available computing power, 
  number of generations/epochs, dataset sizes, etc.). By default, all available CPU cores will be used, but this can
  be customised (see below).

## Installing dependencies

Project dependencies are specified in `pyproject.toml`, and can be installed using [Poetry](https://python-poetry.org/)
(installation instructions [here](https://python-poetry.org/docs/#installation)), using:

```bash
cd evolutionary-exps
poetry install
```

After dependencies have been installed, activate the generated Python virtual environment:

```bash
poetry shell
```

## Running the experiments

All experiments presented in the paper can be replicated running the following (NOTE: this may take a long time!):

```bash
run_evol_coach_exp

# or, equivalently, like this:
python evolutionary_exps/evolutionary_coach.py
```

Alternatively, individual experiments may be run by specifying a list of one or more KB names (the specified names must
be present in the `kbs.json` file in the data directory, see below), e.g.:

```bash
# use spaces to separate KB names
run_evol_coach_exp kb_20_10_10_2 kb_20_9_11_2
```

### Using custom input data

To use custom data, a source data directory can be specified (the directory must have the structure 
described [here](https://github.com/VMarkos/proxy-coaching-argml-2022/blob/main/tree-exps#execution)):

```bash
# make sure to use an absolute path
run_evol_coach_exp --data-dir-path /home/user/Downloads/data
```

### Results

By default, results will be written at `evolutionary-exps/results`, but this can also be customised:

```bash
# make sure to use an absolute path
run_evol_coach_exp --results-dir-path /home/user/Downloads/results
```

### Other options

All other experimental parameters can be customised, running `run_evol_coach_exp --help` shows all available options:

```terminal
 Usage: run_evol_coach_exp [OPTIONS] [KB_NAMES]...                                                                                                   
                                                                                                                                                     
 Allows running evolutionary coach experiments using one or more target KBs.                                                                         
                                                                                                                                                     
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   kb_names      [KB_NAMES]...  Names of the target KBs to train with (use spaces to separate multiple names). If not specified, ALL KBs used in   │
│                                the experiments presented in the paper will be used. If a custom --data-dir-path is specified, all KBs specified   │
│                                in the `kbs.json` file will be used.                                                                               │
│                                [default: None]                                                                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --generations                                            INTEGER                          Number of generations to train for. [default: 100]      │
│ --epochs                                                 INTEGER                          Number of epochs (intervals of generations that start   │
│                                                                                           with the addition of a new rule, and end just before    │
│                                                                                           such an addition) to train for. If provided, the        │
│                                                                                           --generations argument is ignored.                      │
│                                                                                           [default: None]                                         │
│ --t                                                      INTEGER                          Threshold value used to split a population into         │
│                                                                                           beneficial, neutral, and detrimental groups, (see       │
│                                                                                           Valiant, 2009: Evolvability, Journal of the ACM, 56(1), │
│                                                                                           1–21, https://doi.org/10.1145/1462153.1462156)          │
│                                                                                           [default: 0]                                            │
│ --k                                                      INTEGER                          Exponent value used to give higher probability to       │
│                                                                                           organisms with higher fitness during the random         │
│                                                                                           selection from the beneficial group (if the group is    │
│                                                                                           not empty).                                             │
│                                                                                           [default: 2]                                            │
│ --training-set-size-limit                                INTEGER                          Allows limiting the size of the training set.           │
│                                                                                           [default: None]                                         │
│ --data-dir-path                                          PATH                             Allows changing the default directory containing the    │
│                                                                                           data required for the experiments. MUST be an absolute  │
│                                                                                           path.                                                   │
│                                                                                           [default: None]                                         │
│ --results-dir-path                                       PATH                             Allows changing the default directory where the         │
│                                                                                           experiment results will be written. MUST be an absolute │
│                                                                                           path.                                                   │
│                                                                                           [default: None]                                         │
│ --use-multiprocessing        --no-use-multiprocessing                                     Whether to use multiple CPU cores during training       │
│                                                                                           (recommended, since it significantly improves training  │
│                                                                                           speeds). Number of cores to be used can be specified    │
│                                                                                           using --number-of-processes.                            │
│                                                                                           [default: use-multiprocessing]                          │
│ --number-of-processes                                    INTEGER                          Number of cores to use during training. If None, all    │
│                                                                                           available cores will be used. Ignored if                │
│                                                                                           --no-use-multiprocessing is specified.                  │
│                                                                                           [default: None]                                         │
│ --install-completion                                     [bash|zsh|fish|powershell|pwsh]  Install completion for the specified shell.             │
│                                                                                           [default: None]                                         │
│ --show-completion                                        [bash|zsh|fish|powershell|pwsh]  Show completion for the specified shell, to copy it or  │
│                                                                                           customize the installation.                             │
│                                                                                           [default: None]                                         │
│ --help                                                                                    Show this message and exit.                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
