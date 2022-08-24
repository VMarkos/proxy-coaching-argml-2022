import gzip
import json
import multiprocessing as mp
import random
import time
from collections import namedtuple
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd
import typer
from tqdm import tqdm

from evolutionary_exps.prudens_wrappers import (
    to_prudens_literal,
    PrudensRule,
    simplify_prudens_rule,
)
from evolutionary_exps.symbolic_module import PrudensSymbolicModule
from evolutionary_exps.utils import negate, load_datasets_for_kb, get_project_root

# this is for the cli interface
app = typer.Typer()


@app.command()
def run_evolutionary_coach_experiments_cli(
    kb_names: List[str] = typer.Argument(
        None,
        help="Names of the target KBs to train with (use spaces to separate multiple names). If not specified, "
        "ALL KBs used in the experiments presented in the paper will be used. If a custom --data-dir-path is "
        "specified, all KBs specified in the `kbs.json` file will be used.",
    ),
    generations: int = typer.Option(100, help="Number of generations to train for."),
    epochs: int = typer.Option(
        None,
        help="Number of epochs (intervals of generations that start with the addition of a new rule, and end just "
        "before such an addition) to train for. If provided, the --generations argument is ignored.",
    ),
    t: int = typer.Option(
        0,
        help="Threshold value used to split a population into beneficial, neutral, and detrimental groups, (see "
        "Valiant, 2009: Evolvability, Journal of the ACM, 56(1), 1–21, https://doi.org/10.1145/1462153.1462156)",
    ),
    k: int = typer.Option(
        2,
        help="Exponent value used to give higher probability to organisms with higher fitness "
        "during the random selection from the beneficial group (if the group is not empty).",
    ),
    training_set_size_limit: int = typer.Option(
        None, help="Allows limiting the size of the training set."
    ),
    data_dir_path: Path = typer.Option(
        None,
        help="Allows changing the default directory containing the data required for the experiments. MUST be an "
        "absolute path.",
        exists=True,
    ),
    results_dir_path: Path = typer.Option(
        None,
        help="Allows changing the default directory where the experiment results will be written. MUST be an "
        "absolute path.",
    ),
    use_multiprocessing: bool = typer.Option(
        True,
        help="Whether to use multiple CPU cores during training (recommended, since it significantly improves training "
        "speeds). Number of cores to be used can be specified using --number-of-processes.",
    ),
    number_of_processes: int = typer.Option(
        None,
        help="Number of cores to use during training. If None, all available cores will be used. Ignored if "
        "--no-use-multiprocessing is specified.",
    ),
):
    """
    Allows running evolutionary coach experiments using one or more target KBs.
    """
    if data_dir_path is None:
        # if not specified, get data path in relation to this file
        data_dir_path = get_project_root() / "data-gen" / "used-data"
    else:
        assert data_dir_path.is_absolute(), "Please provide an absolute data dir path!"

    if results_dir_path is None:
        # if not specified, use default location for results
        results_dir_path = get_project_root() / "evolutionary-exps" / "results"
    else:
        assert (
            results_dir_path.is_absolute()
        ), "Please provide an absolute results dir path!"

    if not kb_names:
        # if not specified, run experiments for all KBs specified in data_dir_path
        with Path(data_dir_path, "kbs.json").open("r") as f:
            kb_names = json.load(f)

    if len(kb_names) > 1:
        print(f"Running experiments for a total of {len(kb_names)} KBs.\n")

    for kb_name in kb_names:
        run_evolutionary_coach_experiment(
            kb_name=kb_name,
            generations=generations,
            epochs=epochs,
            t=t,
            k=k,
            training_set_size_limit=training_set_size_limit,
            data_dir_path=data_dir_path,
            results_dir_path=results_dir_path,
            use_multiprocessing=use_multiprocessing,
            number_of_processes=number_of_processes,
        )


def run_evolutionary_coach_experiment(
    kb_name: str,
    generations: int = 100,
    epochs: int = None,
    t: int = 0,
    k: int = 2,
    training_set_size_limit: int = None,
    data_dir_path: Path = None,
    results_dir_path: Path = Path(get_project_root(), "evolutionary-exps", "results"),
    use_multiprocessing: bool = True,
    number_of_processes: int = None,
):

    t: threshold, à la Valiant
    k: exponent for weights in random survivor selection
    """

    tz = ZoneInfo("Europe/Nicosia")
    exp_start_time = datetime.now(tz=tz)
    time_format = "%Y-%m-%d %H:%M:%S"

    general_timer_start = time.perf_counter()
    total_deduction_time = 0.0
    total_deductions = 0
    current_population: list[PrudensSymbolicModule] = []
    survivor_history: dict[int, PrudensSymbolicModule] = dict()

    # note the filtering of unlabelled instances from training and testing sets
    training_set, testing_set, coaching_set = load_datasets_for_kb(
        kb_name=kb_name, exclude_unlabelled=True, data_dir_path=data_dir_path
    )

    iterations_number = epochs if epochs is not None else generations
    iterations_type = "epochs" if epochs is not None else "generations"
    iterations_type_secondary = "generations" if epochs is not None else "epochs"

    print("Starting experiment at:", exp_start_time.strftime(time_format))
    print(
        pd.DataFrame.from_dict(
            {
                "KB name": kb_name,
                "Generations": generations if epochs is None else "-",
                "Epochs": epochs or "-",
                "Multiprocessing": use_multiprocessing,
            },
            orient="index",
        ).to_markdown(
            tablefmt="fancy_outline",
            headers=["Experiment parameters", "Value"],
        )
    )

    # prepare all training contexts and labels by converting them to the Prudens literal format
    training_contexts = [
        [to_prudens_literal(c) for c in [*instance.data, "true"]]
        for instance in training_set
    ]
    training_labels: list[str] = [i.label[0] for i in training_set]

    # if requested, sample a random subgroup of the training set
    if training_set_size_limit:
        training_contexts_sample = training_contexts[:training_set_size_limit]
        training_labels_sample = training_labels[:training_set_size_limit]
    else:
        training_contexts_sample = training_contexts
        training_labels_sample = training_labels

    len_training_set_full = len(training_set)
    len_training_set_sample = len(training_contexts_sample)

    # prepare all testing contexts and labels by converting them to the Prudens literal format
    testing_contexts = [
        [to_prudens_literal(c) for c in [*instance.data, "true"]]
        for instance in testing_set
    ]
    testing_labels: list[str] = [i.label[0] for i in testing_set]
    len_testing_set_full = len(testing_set)

    assert (
        len(coaching_set) >= generations
    ), "Generation number larger than available coaching instances!"

    # create an iterator for the coaching set, so it can be sampled one instance at a time
    coaching_set_iterator = iter(coaching_set)

    # get the form of the target label, to use in the generation of new rules
    target_label = str(training_set[0].label[0]).replace("-", "")
    possible_rule_heads = [target_label, negate(target_label)]

    with mp.Pool(processes=number_of_processes) as pool:
        # start training here
        with tqdm(total=iterations_number, desc=iterations_type.title()) as pbar:
            generation = 0
            # using a "while" loop instead of a "for" loop here, to support both a generation or epoch goal
            while True:
                # here we are "cheating" a bit, in the sense that we are sampling a coaching context that will be used
                # to create offspring with new rules in the NEXT generation, not the CURRENT gen - this is done since
                # the label of the coaching context has to be deduced before creating new rules (see below), and that
                # would require calling Prudens for a single deduction once every generation, which slows things down;
                # instead, the coaching context is bundled with the training contexts to be deduced by all organisms of
                # the CURRENT gen, and thus its deduced label will be available in the NEXT gen to create rules
                coaching_context = next(coaching_set_iterator)
                coaching_context_json = [
                    to_prudens_literal(c) for c in [*coaching_context.data, "true"]
                ]

                if generation == 0:
                    # at gen 0, start with a single organism with an empty KB
                    current_population.append(PrudensSymbolicModule())
                else:
                    assert len(current_population) == 1
                    parent = current_population[0]
                    current_population.clear()

                    # create population for this generation:

                    # FIRST, clone parent
                    parent_clone = parent.clone()

                    if parent_clone.kb.is_not_empty():
                        # duplicate the clone's last rule, so rule numbers correspond with gen numbers (this does not
                        # influence fitness)
                        last_rule = parent_clone.kb.rules[-1]
                        parent_clone.induce(
                            PrudensRule(
                                body=last_rule.body,
                                head=last_rule.head,
                                added_as="Clone",
                            ),
                            deactivate_previous_last_rule=True,
                        )

                    # SECOND, using a context from the coaching set (and its deduced label, the operation is done in
                    # the previous gen), create 1 or 2 offspring, each with a new rule that maximally covers the
                    # context (same context and positive or negative head, depending on the deduced label)
                    assert parent.next_gen_coaching_context is not None
                    assert parent.coaching_context_deduction_results is not None

                    possible_rule_heads_temp = possible_rule_heads.copy()
                    if parent.coaching_context_deduction_results:
                        # if organism already has rule that has a prediction for the coaching context, create only
                        # one rule with the opposite of the prediction as the head, otherwise create both possible rules
                        possible_rule_heads_temp.remove(
                            parent.coaching_context_deduction_results[0]
                        )

                    offspring_with_new_rule = []
                    for possible_head in possible_rule_heads_temp:
                        new_rule_based_on_random_context = PrudensRule(
                            body=tuple(parent.next_gen_coaching_context.data),
                            head=possible_head,
                            added_as="Add",
                        )
                        o = parent.clone()
                        o.induce(new_rule_based_on_random_context)
                        offspring_with_new_rule.append(o)

                    # THIRD, create all possible offspring with -1 literal simplified versions of the parent's last rule
                    simplified_last_rule_offspring = []
                    if parent.kb.is_not_empty():
                        simplified_rules = simplify_prudens_rule(parent.kb.rules[-1])
                        for simplified_rule in simplified_rules:
                            offspring = parent.clone()
                            offspring.induce(
                                simplified_rule, deactivate_previous_last_rule=True
                            )
                            simplified_last_rule_offspring.append(offspring)

                    current_population = [
                        parent_clone,
                        *offspring_with_new_rule,
                        *simplified_last_rule_offspring,
                    ]

                # test population here
                if not use_multiprocessing:
                    raise NotImplementedError
                    # prudens_inputs_dict = {
                    #     "kbs": [s.kb.to_string() for s in current_population],
                    #     "data": [*training_contexts_sample, coaching_context_str],
                    # }
                    # deduction_start = time.perf_counter()
                    # prudens_results = run_prudens_alt(prudens_inputs_dict)
                    # total_deduction_time += time.perf_counter() - deduction_start
                    # total_deductions += len(current_population) * len(
                    #     training_contexts_sample
                    # )
                    #
                    # results_for_data, results_for_next_gen_context = [], []
                    # for result_list in prudens_results:
                    #     results_for_data.append(result_list[:-1])
                    #     results_for_next_gen_context.append(result_list[-1])
                    # assert len(results_for_data) == len(current_population)
                    # assert len(results_for_next_gen_context) == len(current_population)
                    #
                    # results_for_all: list[list[int]] = []
                    # for results_for_symoid in results_for_data:
                    #     res_sublist = []
                    #     for output_list, expected in zip(
                    #         results_for_symoid, training_labels_sample
                    #     ):
                    #         if expected in output_list:
                    #             res_sublist.append(1)  # correct
                    #         elif negate(expected) in output_list:
                    #             res_sublist.append(-1)  # wrong
                    #         else:
                    #             res_sublist.append(0)  # abstention
                    #     results_for_all.append(res_sublist)
                    #
                    # for res, s, coach_ctx_res in zip(
                    #     results_for_all,
                    #     current_population,
                    #     results_for_next_gen_context,
                    # ):
                    #     s.set_fitness_scores(fitness_scores=res, gen=generation)
                    #     s.next_gen_coaching_context = coaching_context
                    #     s.coaching_context_deduction_results = coach_ctx_res

                else:
                    deduction_start = time.perf_counter()
                    results = pool.starmap(
                        calc_symbolic_module_perf,
                        tqdm(
                            zip(
                                current_population,
                                repeat(training_contexts_sample),
                                repeat(training_labels_sample),
                                repeat(coaching_context_json),
                            ),
                            total=len(current_population),
                            leave=False,
                            desc="Organisms",
                        ),
                    )
                    total_deduction_time += time.perf_counter() - deduction_start
                    total_deductions += len(current_population) * (
                        len_training_set_sample
                        + 1  # plus 1 for the coaching context deduction
                    )

                    for (s_copy, res, coaching_context_res), s in zip(
                        results, current_population
                    ):
                        # pool.starmap returns ordered results, but the returned organisms are copies
                        assert s_copy.id == s.id
                        s.set_fitness_scores(fitness_scores=res, gen=generation)
                        s.next_gen_coaching_context = coaching_context
                        s.coaching_context_deduction_results = coaching_context_res

                # pick survivor for next gen
                beneficial, neutral, detrimental = [], [], []
                for s in current_population:
                    rel_fitness = s.get_relative_fitness()
                    if rel_fitness > t:
                        beneficial.append(s)
                    elif rel_fitness < -t:
                        detrimental.append(s)
                    else:
                        neutral.append(s)

                if beneficial:
                    survivor = random.choices(
                        population=beneficial,
                        weights=[s.get_relative_fitness() ** k for s in beneficial],
                        k=1,  # this k is for the number of choices
                    )[0]
                elif neutral:
                    survivor = random.choice(neutral)
                else:
                    # by definition, the neutral group will be non-empty (the parent clone at least will have a
                    # relative fitness of 0), and so no organism from the detrimental group will ever be selected
                    raise ValueError(
                        "No organisms found in either beneficial or neutral groups!"
                    )

                # # the below can be used to always choose the top performer as the survivor
                # relative_fitnesses_of_current: list[tuple[int, UUID]] = [
                #     (s.get_relative_fitness(), s.id) for s in current_population
                # ]
                # relative_fitnesses_of_current.sort(reverse=True)
                # survivor = population_database[relative_fitnesses_of_current[0][1]]

                survivor_history[generation] = survivor

                generation += 1
                epoch = survivor.kb.number_of_active_rules()

                pbar.set_postfix_str(
                    f"Max fitness: {survivor.get_personal_fitness()}/{len_training_set_sample}, "
                    f"{iterations_type_secondary.title()}: {epoch if epochs is None else generation}"
                )

                current_population.clear()
                current_population.append(survivor)

                if epochs is None:
                    pbar.update()
                    if generation >= generations:
                        break  # finished
                else:
                    if pbar.n != epoch:
                        # only update when epoch changes, to get accurate time estimations
                        pbar.n = epoch
                        pbar.refresh()
                    if epoch >= epochs:
                        break  # finished

    # here test all survivors on testing and coaching sets (and training set, only if a subset was used during training)
    stats_json = {}
    other_info_json = {}
    all_survivors = list(survivor_history.values())
    stats_json["survivor_kb_lengths"] = [
        s.kb.number_of_active_rules() for s in all_survivors
    ]

    # process coaching dataset
    coaching_set_original = coaching_set[:500]
    coaching_set_original = [i for i in coaching_set_original if i.label]
    coaching_contexts = [
        [to_prudens_literal(c) for c in [*instance.data, "true"]]
        for instance in coaching_set_original
    ]
    coaching_labels: list[str] = [i.label[0] for i in coaching_set_original]

    Dt = namedtuple("Dt", ["type", "contexts", "labels"])
    all_dts = [
        Dt("train", training_contexts, training_labels),
        Dt("test", testing_contexts, testing_labels),
        # Dt("coach", coaching_contexts, coaching_labels),
    ]

    with mp.Pool(processes=number_of_processes) as pool:

        for dt in all_dts:
            deduction_start = time.perf_counter()
            dt_results = pool.starmap(
                calc_symbolic_module_perf,
                tqdm(
                    zip(all_survivors, repeat(dt.contexts), repeat(dt.labels)),
                    total=len(all_survivors),
                    leave=True,
                    desc=f"Evaluating full {dt.type} set",
                ),
            )
            total_deduction_time += time.perf_counter() - deduction_start
            total_deductions += len(all_survivors) * len(dt.contexts)

            survivor_predictions: list[list[int]] = [r[1] for r in dt_results]

            total_count = len(dt.contexts)
            stats_json[dt.type] = {
                "correct": [p.count(1) / total_count for p in survivor_predictions],
                "correct_count": [p.count(1) for p in survivor_predictions],
                "abstain": [p.count(0) / total_count for p in survivor_predictions],
                "abstain_count": [p.count(0) for p in survivor_predictions],
                "wrong": [p.count(-1) / total_count for p in survivor_predictions],
                "wrong_count": [p.count(-1) for p in survivor_predictions],
                "personal": [sum(p) for p in survivor_predictions],
            }

    total_time = time.perf_counter() - general_timer_start

    other_info = {
        "KB name": kb_name,
        "Training set length": len_training_set_full,
        "Training subsampling": training_set_size_limit or "No subsampling",
        "Testing set length": len_testing_set_full,
        "Fitness - train": stats_json["train"]["personal"][-1],
        "Fitness - test": stats_json["test"]["personal"][-1],
        # "Fitness - coach": stats_json["coach"]["personal"][-1],
        "Total time (s)": round(total_time, 2),
        "Python time (s)": round(total_time - total_deduction_time, 2),
        "Deduction time (s)": round(total_deduction_time, 2),
        "Total deductions": total_deductions,
        "ms/deduction": round((total_deduction_time / total_deductions) * 1000, 4),
    }

    other_info_json["other_info"] = other_info

    other_info_json["survivors_per_gen"] = {
        k: str(v.id) for k, v in survivor_history.items()
    }

    all_organisms = {}
    for survivor in all_survivors:
        all_organisms[str(survivor.id)] = {
            "kb": survivor.kb.to_string(),
            "lineage": [str(i) for i in survivor.lineage],
            "fitness_scores": survivor.get_fitness_scores(),
            "personal_fitness": survivor.get_personal_fitness(),
            "relative_fitness": int(survivor.get_relative_fitness()),
            "fitness_by_rule_stats": survivor.get_fitness_by_rule_stats(),
        }

    other_info_json["all_organisms"] = all_organisms

    exp_results_dir_path = Path(
        results_dir_path, f"{kb_name}_{exp_start_time.strftime('%Y%m%d%H%M%S')}"
    )
    exp_results_dir_path.mkdir(exist_ok=True, parents=True)

    with Path(exp_results_dir_path, f"{kb_name}.json").open("w+") as f:
        json.dump(stats_json, f)

    with gzip.open(
        Path(exp_results_dir_path, f"{kb_name}-other_info.json.gz"), "w"
    ) as f:
        f.write(json.dumps(other_info_json).encode("utf-8"))

    del (
        training_set,
        testing_set,
        coaching_set,
        current_population,
        survivor_history,
        training_contexts,
        training_labels,
        stats_json,
        other_info_json,
        all_organisms,
    )

    print("Results saved at:", exp_results_dir_path)
    print(
        pd.DataFrame.from_dict(other_info, orient="index").to_markdown(
            tablefmt="fancy_outline",
            headers=["Experiment results", "Value"],
        )
    )
    print("Finished at:", datetime.now(tz=tz).strftime(time_format), "\n")


def calc_symbolic_module_perf(
    sym_mod: PrudensSymbolicModule,
    context_strings: list[list],
    expected_labels: list[str],
    next_gen_coaching_context: str = None,
) -> tuple[PrudensSymbolicModule, list[int], list]:
    res_for_next_gen_coaching_context = []
    if next_gen_coaching_context is not None:
        # include coaching context of the following generation in this gen's deductions, to reduce the number of
        # interactions with Prudens
        outputs = sym_mod.deduce(
            side_input=[], theory_input=[*context_strings, next_gen_coaching_context]
        )
        res_for_next_gen_coaching_context = outputs.pop()
        assert len(outputs) == len(expected_labels)
    else:
        outputs = sym_mod.deduce(side_input=[], theory_input=context_strings)
        assert len(outputs) == len(expected_labels)

    results = []
    for output_list, expected in zip(outputs, expected_labels):
        if expected in output_list:
            results.append(1)  # correct prediction
        elif negate(expected) in output_list:
            results.append(-1)  # wrong prediction
        else:
            results.append(0)  # abstention
    return sym_mod, results, res_for_next_gen_coaching_context


if __name__ == "__main__":
    app()
