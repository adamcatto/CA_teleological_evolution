"""
Rule-set validators and small soft bonuses.

The original code used giant negative penalties (-100k / -10k) to discourage
"reward-hacking" rules (e.g. rules that conjure truth values out of thin air). Those
penalties dominated gradients. The cleaner pattern is to expose **validators** that
return the indices of offending rules, and let the policy layer either:
  (a) mask those rules out before they enter the CA simulation (preferred), or
  (b) penalize them with a *bounded* per-violation cost.

The desirable-property check (`has_center_output_to_truth_rule`) is kept as a
small soft bonus the composer can apply.
"""
from typing import List

import numpy as np


__all__ = [
    "find_truth_value_violators",
    "find_output_neuron_creation_violators",
    "rule_violation_indices",
    "has_center_output_to_truth_rules",
]


def _as_rules(rules_or_ruleset):
    """Accept a RuleSet, list of Rule, or any iterable of Rule."""
    return list(rules_or_ruleset)


def _truth_value_set(vocabulary):
    return {vocabulary.TRUE.value, vocabulary.FALSE.value, vocabulary.INTERMEDIATE_TRUTH_VALUE.value}


def find_truth_value_violators(rules_or_ruleset, vocabulary) -> List[int]:
    """
    Indices of rules that conjure a truth-value token in the output without any truth
    value present in the input. These are the classic terminal-reward-hacking pattern.
    """
    rules = _as_rules(rules_or_ruleset)
    tvs = _truth_value_set(vocabulary)
    out = []
    for i, r in enumerate(rules):
        if np.isin(r.output_grid, list(tvs)).any() and not np.isin(r.input_grid, list(tvs)).any():
            out.append(i)
    return out


def find_output_neuron_creation_violators(rules_or_ruleset, vocabulary) -> List[int]:
    """
    Indices of rules that introduce an OUTPUT_NEURON token in the output without one
    in the input — these would "teleport" the output neuron, breaking the input→output
    mapping the system is trying to learn.
    """
    rules = _as_rules(rules_or_ruleset)
    on = vocabulary.OUTPUT_NEURON.value
    out = []
    for i, r in enumerate(rules):
        if np.isin(r.output_grid, on).any() and not np.isin(r.input_grid, on).any():
            out.append(i)
    return out


def rule_violation_indices(rules_or_ruleset, vocabulary) -> List[int]:
    """Union of all hard-constraint violators."""
    a = set(find_truth_value_violators(rules_or_ruleset, vocabulary))
    b = set(find_output_neuron_creation_violators(rules_or_ruleset, vocabulary))
    return sorted(a | b)


def has_center_output_to_truth_rules(rules_or_ruleset, vocabulary):
    """
    Returns (has_true_rule, has_false_rule) — whether the rule set contains a rule
    that maps OUTPUT_NEURON in the center of the input to TRUE/FALSE in the center
    of the output. Both being present is necessary for the CA to be able to ever
    produce both polarities of XOR output.
    """
    rules = _as_rules(rules_or_ruleset)
    on = vocabulary.OUTPUT_NEURON.value
    has_true = False
    has_false = False
    for r in rules:
        if r.input_grid[1, 1] == on:
            if r.output_grid[1, 1] == vocabulary.TRUE.value:
                has_true = True
            elif r.output_grid[1, 1] == vocabulary.FALSE.value:
                has_false = True
    return has_true, has_false
