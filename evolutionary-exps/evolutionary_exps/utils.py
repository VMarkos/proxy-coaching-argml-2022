from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Instance(object):
    """
    A class to represent a single instance of training data.
    """

    data: list
    side: list
    label: list


def negate(variable: str) -> str:
    """
    Negates the provided variable.
    """
    if variable.startswith("-"):
        return variable.lstrip("-")
    else:
        return str(f"-{variable}")


def maybe_negate(variable: str, negation_probability: float = 0.5) -> str:
    """
    Negates the provided variable based on the negation probability.

    :param variable: The variable to maybe negate
    :param negation_probability: The probability of negation - 0.0 will not negate any variables, 1.0 will
     negate all variables
    """
    assert (
        0.0 <= negation_probability <= 1.0
    ), "Negation probability must be in the range [0.0, 1.0]"

    if random.choices(
        [True, False],
        weights=[negation_probability, 1.0 - negation_probability],
        k=1,
    )[0]:
        return negate(variable)
    else:
        return variable


def get_project_root() -> Path:
    """
    Returns the root folder of the project.
    """
    return Path(__file__).absolute().parent.parent.parent


def load_datasets_for_kb(
    kb_name: str, data_dir_path: Path, exclude_unlabelled: bool = True
) -> tuple[list[Instance], list[Instance], list[Instance]]:
    """
    Loads datasets for the specified knowledge base/policy.

    :param kb_name: name of the kb/policy
    :param data_dir_path: path to the directory containing data files
    :param exclude_unlabelled: whether to filter out unlabelled data instances (NOTE: this only applies to the
     training and testing sets, NOT the coaching set)
    """
    assert data_dir_path.is_dir()

    with Path(data_dir_path, "kbs.json").open("r") as f:
        all_kb_names: list[str] = json.load(f)
        assert kb_name in all_kb_names, f"{kb_name=} not valid."

    # read raw data from json files
    with Path(data_dir_path, "trainTest.json").open("r") as f:
        train_test_sets = json.load(f)
        train_raw = train_test_sets[kb_name]["train"]
        test_raw = train_test_sets[kb_name]["test"]

    with Path(data_dir_path, "coaching.json").open("r") as f:
        coach_raw = json.load(f)[kb_name]

    # convert sets using the Instance class, for safer access to the data
    conv_train_test_sets = []
    for raw_set in [train_raw, test_raw]:
        conv_set = [
            Instance(
                data=i["context"],
                side=[],
                label=[i["label"]] if i["label"] is not None else [],
            )
            for i in raw_set
        ]

        if exclude_unlabelled:
            # if requested, filter out unlabelled instances (ONLY for training and testing)
            conv_set = [i for i in conv_set if i.label]

        conv_train_test_sets.append(conv_set)

    training_set, testing_set = conv_train_test_sets

    # NOTE that no label-based filtering is applied to the coaching set
    coaching_set = [Instance(data=i, side=[], label=[]) for i in coach_raw["full"]]

    return training_set, testing_set, coaching_set
