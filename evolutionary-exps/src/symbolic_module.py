from abc import ABC, abstractmethod
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from prudens_wrappers import (
    PrudensRule,
    PrudensKnowledgeBase,
    run_prudens,
    get_prudens_source,
)


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
    def induce(self, new_rule, **kwargs):
        pass


class PrudensSymbolicModule(SymbolicModule):
    """
    Implementation of SymbolicModule using Prudens (https://github.com/VMarkos/prudens-js) as the underlying engine.
    """

    def __init__(
        self,
        *,
        inherited_knowledge_base: PrudensKnowledgeBase = None,
        inherited_fitness_scores: dict[int, list[int]] = None,
        lineage: list[UUID] = None,
    ) -> None:
        self.id = uuid4()

        self._prudens_source_file = get_prudens_source(
            sha="eb5b6f09224828f5f1eaa9bc956ceb8f8a27d7cb"
        )

        self.kb = (
            PrudensKnowledgeBase(
                rules=(PrudensRule(body=("empty",), head="empty_rule"),)
            )
            if inherited_knowledge_base is None
            else PrudensKnowledgeBase(
                rules=tuple(r for r in inherited_knowledge_base.rules)
            )
        )

        self._inherited_fitness_scores: dict[int, list[int]] = (
            {} if inherited_fitness_scores is None else inherited_fitness_scores.copy()
        )

        self._personal_fitness_scores: list[int] = []

        self.lineage: list[UUID] = [] if lineage is None else [a for a in lineage]

        self.next_gen_coaching_context = None
        self.coaching_context_deduction_results = None

    def deduce(self, side_input: list, theory_input: list[list]) -> list[list]:
        """
        Performs Prudens deduction using the current knowledge base and the provided inputs.
        """
        if self.kb.is_empty():
            # if kb is empty, it essentially means that it will always abstain (also Prudens gives errors when run
            # with emtpy kb)
            return [[] for _ in theory_input]

        prudens_inputs_dict = {
            "kbs": [self.kb.to_prudens_kb_json_object()],
            "data": theory_input,
        }
        results = run_prudens(
            prudens_inputs=prudens_inputs_dict,
            source_file_path=self._prudens_source_file,
        )

        assert len(results) == 1
        results = results[0]
        assert len(theory_input) == len(results), (
            f"Length of theory inputs ({len(theory_input)}) "
            f"does not match the length of results ({len(results)})!"
        )
        return results

    def induce(
        self, new_rule: PrudensRule, deactivate_previous_last_rule: bool = False
    ):
        """
        Performs induction as described in the Machine Coaching paper (Michael 2019)
        (https://cognition.ouc.ac.cy/loizos/papers/Michael_2019_MachineCoaching.pdf).
        """
        # here replace the whole KB with new rule appended
        if deactivate_previous_last_rule:
            last_rule = self.kb.rules[-1]
            deactivated_last_rule = PrudensRule(
                body=last_rule.body,
                head=last_rule.head,
                added_as=last_rule.added_as,
                active=False,
            )
            self.kb = PrudensKnowledgeBase(
                rules=(*self.kb.rules[:-1], deactivated_last_rule, new_rule)
            )
        else:
            self.kb = PrudensKnowledgeBase(rules=(*self.kb.rules, new_rule))

    def clone(self) -> "PrudensSymbolicModule":
        return PrudensSymbolicModule(
            inherited_knowledge_base=self.kb,
            inherited_fitness_scores=self._inherited_fitness_scores,
            lineage=[*self.lineage, self.id],
        )

    def set_fitness_scores(self, fitness_scores: list[int], gen: int):
        assert (
            gen not in self._inherited_fitness_scores
        ), "You are trying to assign fitness_scores to a previous generation!"
        self._inherited_fitness_scores[gen] = fitness_scores
        self._personal_fitness_scores = fitness_scores

    def get_fitness_scores(self):
        return self._personal_fitness_scores

    def get_personal_fitness(
        self,
        # fitness_measure: FitnessMeasure = FitnessMeasure.STRICT,
    ) -> int:
        if not self._personal_fitness_scores:
            raise ValueError("Organism fitness_scores have not been set yet!")

        # if fitness_measure is FitnessMeasure.SIMPLE:
        #     # correct predictions give +1, abstentions and wrong predictions +0
        #     fitness = sum(r for r in self._personal_fitness_scores if r == 1)
        # elif fitness_measure is FitnessMeasure.STRICT:
        #     # correct predictions give +1, abstentions +0 and wrong predictions -1
        #     fitness = sum(self._personal_fitness_scores)
        # else:
        #     raise ValueError(f"fitness_measure {fitness_measure} is not a valid value!")
        #
        # return fitness

        # personal fitness is the sum of _personal_fitness_scores, at the only generation the organism lived
        # correct predictions give +1, abstentions +0 and wrong predictions -1
        return sum(self._personal_fitness_scores)

    def get_relative_fitness(self) -> int:
        if not self._personal_fitness_scores:
            raise ValueError("Organism fitness_scores have not been set yet!")
        if len(self._inherited_fitness_scores) == 1:
            if len(self.kb.rules) == 1:
                # this means that this organism is the single ancestor with a single empty rule
                assert self.kb.rules[0] == PrudensRule(
                    body=("empty",), head="empty_rule"
                )
                fitness = sum(self._personal_fitness_scores)
                assert fitness == 0
                return fitness
            else:
                raise ValueError(
                    "No information about previous inherited_fitness_scores is available!"
                )
        parent_fitness_scores = self._inherited_fitness_scores[
            # parent is the second before last item in inherited scores, since personal scores are automatically added
            list(self._inherited_fitness_scores.keys())[-2]
        ]
        assert len(parent_fitness_scores) == len(self._personal_fitness_scores)
        pairwise_diffs: np.ndarray = np.array(self._personal_fitness_scores) - np.array(
            parent_fitness_scores
        )
        pairwise_diffs[pairwise_diffs == 2] = 1
        pairwise_diffs[pairwise_diffs == -2] = -1
        return pairwise_diffs.sum()

    def get_fitness_by_rule_stats(self, print_stats: bool = False) -> dict:
        if not self._personal_fitness_scores:
            raise ValueError("Organism fitness_scores have not been set yet!")

        assert len(self._inherited_fitness_scores.keys()) == len(
            self.kb.rules
        ), f"{len(self._inherited_fitness_scores.keys())} != {len(self.kb.rules)}"

        stats = []
        for (gen, scores), rule in zip(
            self._inherited_fitness_scores.items(), self.kb.rules
        ):
            personal = sum(scores)
            if gen == 0:
                relative = 0
            else:
                parent_scores = self._inherited_fitness_scores[gen - 1]
                assert len(parent_scores) == len(scores)
                pairwise_diffs: np.ndarray = np.array(scores) - np.array(parent_scores)
                pairwise_diffs[pairwise_diffs == 2] = 1
                pairwise_diffs[pairwise_diffs == -2] = -1
                relative = pairwise_diffs.sum()

            correct = scores.count(1)
            abstain = scores.count(0)
            wrong = scores.count(-1)
            stats.append(
                [
                    rule.to_string(rule_name=gen),
                    personal,
                    relative,
                    rule.added_as,
                    rule.active,
                    correct,
                    wrong,
                    abstain,
                ]
            )
        stats_df = pd.DataFrame(
            stats,
            columns=[
                "Rule",
                "Personal",
                "Relative",
                "Mutation type",
                "Active",
                "Correct",
                "Wrong",
                "Abstain",
            ],
        )

        if print_stats:
            print(stats_df.to_markdown(index=False))

        return stats_df.to_dict(orient="list")

    def abduce(self, side_input: list, theory_output: list):
        """
        Abduction capabilities are not relevant to the evolutionary coaching experiments, and so this method is not
        implemented.
        """
        raise NotImplementedError
