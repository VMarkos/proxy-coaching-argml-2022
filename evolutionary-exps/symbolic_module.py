from abc import ABC, abstractmethod

from prudens_wrappers import PrudensRule


class Module(ABC):
    """
    Base abstract class for neural and symbolic modules, as proposed by Tsamoura et al., 2021
    (https://ojs.aaai.org/index.php/AAAI/article/view/16639).
    """

    pass


class SymbolicModule(Module):
    """
    Abstract class for the symbolic module in the Neural-Symbolic architecture proposed by Tsamoura et al., 2021
    (https://ojs.aaai.org/index.php/AAAI/article/view/16639).
    """

    @abstractmethod
    def abduce(self, side_input: list, theory_output: list):
        pass

    @abstractmethod
    def deduce(self, side_input: list, theory_input: list):
        pass

    @abstractmethod
    def induce(self, new_rule: PrudensRule):
        pass
