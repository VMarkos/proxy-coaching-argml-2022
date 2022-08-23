import gzip
import json
import multiprocessing as mp
import random
import time
from collections import namedtuple
from datetime import datetime
from itertools import repeat
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

from prudens_wrappers import to_prudens_literal, PrudensRule, simplify_prudens_rule
from symbolic_module import PrudensSymbolicModule
from utils import negate, load_datasets_for_kb


def run_evolutionary_coach_experiment(
    kb_name: str,
    generations: int = 100,
    epochs: int = None,
    use_multiprocessing: bool = True,
    number_of_processes: int = None,
    training_set_size_limit: int = None,
    t: int = 0,
    k: int = 2,
):
    """
    If epochs is provided, the number for generations is ignored.

    t: threshold, à la Valiant
    k: exponent for weights in random survivor selection
    """
    tz, time_format = ZoneInfo("Europe/Nicosia"), "%Y-%m-%d %H:%M:%S"

    general_start = time.perf_counter()
    total_deduction_time = 0.0
    total_deductions = 0
    current_population: list[PrudensSymbolicModule] = []
    survivor_history: dict[int, PrudensSymbolicModule] = dict()

    # note the filtering of unlabelled instances from training and testing sets
    training_set, testing_set, coaching_set = load_datasets_for_kb(
        kb_name, exclude_unlabelled=True
    )

    iterations_number = epochs if epochs is not None else generations
    iterations_type = "epochs" if epochs is not None else "generations"
    iterations_type_secondary = "generations" if epochs is not None else "epochs"

    print(
        f"{datetime.now(tz=tz).strftime(time_format)} - "
        f"Running experiment for '{kb_name}', {iterations_number} {iterations_type}, "
        f"{len(training_set)} training set ({training_set_size_limit or 'no'} random sampling), "
        f"{len(testing_set)} testing set, using {number_of_processes or 'unlimited'} processes."
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
                    # TODO the parent clone should preserve its fitness scores, so it's not tested again
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
                        # pool.starmap returns ordered results, but the returned symoid instance is a copy
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
        Dt("coach", coaching_contexts, coaching_labels),
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

        # deduction_start = time.perf_counter()
        # full_training_dataset_predictions = pool.starmap(
        #     test_symoidv2_preprocessed2,
        #     tqdm(
        #         zip(all_survivors, repeat(training_contexts), repeat(training_labels)),
        #         total=len(all_survivors),
        #         leave=True,
        #         desc="Evaluating full training set",
        #     ),
        # )
        # total_deduction_time += time.perf_counter() - deduction_start
        # total_deductions += len(all_survivors) * len(training_contexts)
        #
        # deduction_start = time.perf_counter()
        # full_testing_dataset_predictions = pool.starmap(
        #     test_symoidv2_preprocessed2,
        #     tqdm(
        #         zip(all_survivors, repeat(testing_contexts), repeat(testing_labels)),
        #         total=len(all_survivors),
        #         leave=True,
        #         desc="Evaluating full testing set",
        #     ),
        # )
        # total_deduction_time += time.perf_counter() - deduction_start
        # total_deductions += len(all_survivors) * len(testing_contexts)

    # assert len(full_training_dataset_predictions) == len(
    #     full_testing_dataset_predictions
    # )

    # train_correct_preds_per_gen, test_correct_preds_per_gen = [], []
    # train_abstain_preds_per_gen, test_abstain_preds_per_gen = [], []
    # train_wrong_preds_per_gen, test_wrong_preds_per_gen = [], []
    # train_personal_fitness_per_gen, test_personal_fitness_per_gen = [], []

    # for (s1, train_res, _), (s2, test_res, _) in zip(
    #     full_training_dataset_predictions, full_testing_dataset_predictions
    # ):
    #     assert s1.id == s2.id
    #
    #     train_correct_preds_per_gen.append(train_res.count(1))
    #     train_abstain_preds_per_gen.append(train_res.count(0))
    #     train_wrong_preds_per_gen.append(train_res.count(-1))
    #     train_personal_fitness_per_gen.append(sum(train_res))
    #
    #     test_correct_preds_per_gen.append(test_res.count(1))
    #     test_abstain_preds_per_gen.append(test_res.count(0))
    #     test_wrong_preds_per_gen.append(test_res.count(-1))
    #     test_personal_fitness_per_gen.append(sum(test_res))

    # training_stats = {
    #     "correct": train_correct_preds_per_gen,
    #     "abstain": train_abstain_preds_per_gen,
    #     "wrong": train_wrong_preds_per_gen,
    #     "personal": train_personal_fitness_per_gen,
    # }
    # testing_stats = {
    #     "correct": test_correct_preds_per_gen,
    #     "abstain": test_abstain_preds_per_gen,
    #     "wrong": test_wrong_preds_per_gen,
    #     "personal": test_personal_fitness_per_gen,
    # }

    # stats_json["training"] = training_stats
    # stats_json["training_percent"] = {
    #     k: [i / len_training_set_full for i in v] for k, v in training_stats.items()
    # }
    # stats_json["testing"] = testing_stats
    # stats_json["testing_percent"] = {
    #     k: [i / len_testing_set_full for i in v] for k, v in testing_stats.items()
    # }

    total_time = time.perf_counter() - general_start

    other_info = {
        "KB name": kb_name,
        "Training set length": len_training_set_full,
        "Training subsampling": training_set_size_limit or "No subsampling",
        "Testing set length": len_testing_set_full,
        "Fitness - train": stats_json["train"]["personal"][-1],
        "Fitness - test": stats_json["test"]["personal"][-1],
        "Fitness - coach": stats_json["coach"]["personal"][-1],
        "Total time (s)": round(total_time, 2),
        "Python time (s)": round(total_time - total_deduction_time, 2),
        "Deduction time (s)": round(total_deduction_time, 2),
        "Total deductions": total_deductions,
        "ms/deduction": round((total_deduction_time / total_deductions) * 1000, 4),
    }

    stats_df = pd.DataFrame.from_dict(other_info, orient="index").transpose()

    print(stats_df.to_markdown(index=False))

    other_info_json["other_info"] = other_info

    other_info_json["survivors_per_gen"] = {
        k: str(v.id) for k, v in survivor_history.items()
    }

    all_organisms = {}
    for symoid in all_survivors:
        all_organisms[str(symoid.id)] = {
            "kb": symoid.kb.to_string(),
            "lineage": [str(i) for i in symoid.lineage],
            "fitness_scores": symoid.get_fitness_scores(),
            "personal_fitness": symoid.get_personal_fitness(),
            "relative_fitness": int(symoid.get_relative_fitness()),
            "fitness_by_rule_stats": symoid.get_fitness_by_rule_stats(),
        }

    other_info_json["all_organisms"] = all_organisms

    results_files_folder = Path(Path.cwd(), "results_files")
    results_files_folder.mkdir(exist_ok=True)

    with Path(results_files_folder, f"{kb_name}.json").open("w+") as f:
        json.dump(stats_json, f)

    with gzip.open(
        Path(results_files_folder, f"{kb_name}-other_info.json.gz"), "w"
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
        stats_df,
        other_info_json,
        all_organisms,
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


def main():
    load_datasets_for_kb("kb_20_2_4_1")


if __name__ == "__main__":
    main()
