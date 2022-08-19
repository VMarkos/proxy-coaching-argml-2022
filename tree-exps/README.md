# Experiments
Notes on `experiments.js`.

## Purpose
`experiments.js` provides any functionality needed to generate data related to the proxy coaching experiments presented in the corresponding paper.

## Dependencies
In order to run the `experiments.js` script, you need to have installed npm - instructions [here](). Furthermore, the script depends on the following scripts to run properly:
* `fs`, which is a native `node.js` script;
* `perf-hooks`, which is a native `node.js` script;
* `parsers.j`, which is a Prudens-related script, found in `prudens-local/parsers.js`;
* `prudens.j`, which is a Prudens-related script, found in `prudens-local/prudens.js`;
* `decision-tree`, which may be found [here](https://www.npmjs.com/package/decision-tree).

Provided that the repository's directory structure is kept intact, there is no further action needed to be taken so as to run `experiments.js`.

## Execution

### Full-path Experiments

In order to run experiments using trained Decision Trees as proxy coaches under the full path advice protocol, `cd` to the directory where `experiments.js` is located and run the following command:

```
npx run-func experiments.js fullPathExps
```

If promted to install `run-func`, press `y` (confirm). 

By default, `fullPathExps` uses all data used in the original experiments, as found in the `/used-data` directory. In case you wish to run experiments with your own data, you may provide the paths to the corresponding files as follows:

```
npx run-func experiments.js fullPathExps "kbIdsPath" "treesPath" "policiesPath" "coachingPath" "trainTestPath"
```

Examples of each file may be found at `/used-data` and, namely:
* `/used-data/kbs.json` for `kbIdsPath`;
*  `/used-data/trainedTrees.json` for `treesPath`;
* `/used-data/policies.json` for `policiesPath`;
* `/used-data/coaching.json` `coachingPath`;
* `/used-data/trainTest.json` for `trainTestPath`.

Any results will be written at `results/fullPathExps.json`.

### Min-prefix Experiments

In order to run experiments using trained Decision Trees as proxy coaches under the minimal prefix advice protocol, `cd` to the directory where `experiments.js` is located and run the following command:

```
npx run-func experiments.js minPrefixExps
```

By default, `minPrefixExps` uses all data used in the original experiments, as found in the `/used-data` directory. In case you wish to run experiments with your own data, you may provide the paths to the corresponding files as follows:

```
npx run-func experiments.js fullPathExps "kbIdsPath" "treesPath" "policiesPath" "coachingPath" "trainTestPath"
```

Examples of each file may be found at `/used-data` and, namely:
* `/used-data/kbs.json` for `kbIdsPath`;
*  `/used-data/trainedTrees.json` for `treesPath`;
* `/used-data/policies.json` for `policiesPath`;
* `/used-data/coaching.json` `coachingPath`;
* `/used-data/trainTest.json` for `trainTestPath`.

Any results will be written at `results/minPrefixExps.json`.

### Forest Experiments

In order to run experiments using trained random forests as proxy coaches, `cd` to the directory where `experiments.js` is located and run the following command:

```
npx run-func experiments.js forestExps
```

If promted to install `run-func`, press `y` (confirm). 

By default, `fullPathExps` uses all data used in the original experiments, as found in the `/used-data` directory. In case you wish to run experiments with your own data, you may provide the paths to the corresponding files as follows:

```
npx run-func experiments.js fullPathExps "forestPath" "kbIdsPath" "policiesPath" "coachingPath" "trainTestPath"
```

Examples of each file may be found at `/used-data` and, namely:
*  `/used-data/trainedForests.json` for `forestPath`;
* `/used-data/kbs.json` for `kbIdsPath`;
* `/used-data/policies.json` for `policiesPath`;
* `/used-data/coaching.json` `coachingPath`;
* `/used-data/trainTest.json` for `trainTestPath`.

Any results will be written at `results/forestExps.json`.
