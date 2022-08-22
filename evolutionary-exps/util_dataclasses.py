from dataclasses import dataclass


@dataclass
class Instance(object):
    """A class to represent a single instance of training data."""

    data: list
    side: list
    label: list
