# Data Generation
Notes on `dataGen.js`.

## Purpose
`dataGen.js` provides any functionality needed to generate data related to the proxy coaching experiments presented in the corresponding paper.

## Dependencies
In order to run the `dataGen.js` script, you need to have installed npm - instructions [here](). Furthermore, the script depends on the following scripts to run properly:
* `fs`, which is a native `node.js` script;
* `perf-hooks`, which is a native `node.js` script;
* `parsers.j`, which is a Prudens-related script, found in `prudens-local/parsers.js`;
* `prudens.j`, which is a Prudens-related script, found in `prudens-local/prudens.js`;
* `decision-tree`, which may be found [here](https://www.npmjs.com/package/decision-tree).

Provided that the repository's directory structure is kept intact, there is no further action needed to be taken so as to run `dataGen.js`.

## Execution

### Policy Generation

In order to generate policies according to the described protocol presented in the paper, you may `cd` to the directory where `dataGen.js` is located and run the following command:

```
npx run-func dataGen.js stackGen
```

If promted to install `run-func`, press `y` (confirm). Any results will be written at `data/policies/genKbs.json`.

**Remark**: Since there are some random choices made during the data generation process, the resulting policies might not be identical to the ones used in the original experiments. The original datasets and any related metadata used may be found at `/used-data`.

### Train-Test Dataset Generation

In order to generate train and test datasets from a set of policies, run the following command:

```
npx run-func dataGen.js generateTrainTest
```

To facilitate reproducibility, by default `generateTrainTest` uses the originally used data found at `/used-data`. However, in case you wish to use other policy sets, you may pass the corresponding paths as parameters, as follows:

```
npx run-func dataGen.js generateTrainTest "policyTagsPath" "policiesPath"
```

In the above `policiesPath` is a string corresponding to the path of policies while `policyTagsPath` is a string corresponding to the path of a file containing the tags of the policies to be used - as in `/used-data/kbs.json`.

Any results will be written to `data/datasets/trainTest.json`.

### Coaching Dataset Generation

In order to generate the coaching dataset from a set of policies, run the following command:

```
npx run-func dataGen.js generateCoaching
```

To facilitate reproducibility, by default `generateCoaching` uses the originally used data found at `/used-data`. However, in case you wish to use other policy sets, you may pass the corresponding paths as parameters, as follows:

```
npx run-func dataGen.js generateTrainTest "n" "policyTagsPath" "policiesPath"
```

In the above, `n` is the number of coaching sets to be generated, `policiesPath` is a string corresponding to the path of policies while `policyTagsPath` is a string corresponding to the path of a file containing the tags of the policies to be used - as in `/used-data/kbs.json`.

Any results will be written to `data/datasets/coaching.json`.

### Proxy Coach Generation

#### Tree Training

To train trees as proxy coaches, run the following command:

```
npx run-func dataGen.js trainTrees
```

As with `generateTrainTest`, `trainTrees` by default uses data used in the original experiments, to facilitate reproducibility. In case you wish to run the experiments with your own data, provide the corresponding paths as parameters, as shown below:

```
npx run-func dataGen.js trainTrees "policyTagsPath" "policiesPath" "trainTestPath"
```

`policyTagsPath` and `policiesPath` are as above, while `trainTestPath` corresponds to the path of a file containing labelled train and test sets, as, e.g., `/used-data/trainTest.json`.

Any results will be written at `data/coaches/trainedTrees.json`.

#### Forest Training

To train random forests as proxy coaches, run the following command:

```
npx run-func dataGen.js trainForests
```

As with `generateTrainTest`, `trainForests` by default uses data used in the original experiments, to facilitate reproducibility. In case you wish to run the experiments with your own data, provide the corresponding paths as parameters, as shown below:

```
npx run-func dataGen.js trainForests "policyTagsPath" "policiesPath" "trainTestPath"
```

`policyTagsPath` and `policiesPath` are as above, while `trainTestPath` corresponds to the path of a file containing labelled train and test sets, as, e.g., `/used-data/trainTest.json`.

Any results will be written at `data/coaches/trainedForests.json`.