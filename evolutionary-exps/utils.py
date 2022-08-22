import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Instance(object):
    """A class to represent a single instance of training data."""

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
    return Path(__file__).absolute().parent.parent
