"""
Episode runners.

`AutomataSession` runs a fixed (non-learning) automaton + rule set against the
shaped reward composer and reports the trajectory's total return + whether the
XOR target was reached.
"""
from .rewards import RewardConfig, compute_init_reward, compute_step_reward, compute_rule_set_score, is_xor_solved


class AutomataSession:
    def __init__(self, automaton, config: RewardConfig = None):
        self.automaton = automaton
        self.config = config if config is not None else RewardConfig()

        self.total_reward = 0.0
        self.evaluates_xor = False
        self._init_reward, self._phi = compute_init_reward(self.automaton, self.config)
        self.total_reward += self._init_reward
        # Rule-set score is paid once at the start of the episode (rules are fixed for the run).
        self.total_reward += compute_rule_set_score(self.automaton.rule_set, self.automaton.vocabulary, self.config)

    def run(self, num_steps: int = None) -> float:
        if num_steps is None:
            num_steps = self.automaton.max_steps
        if num_steps is None or num_steps < 0:
            raise ValueError("num_steps must be a non-negative int (got max_steps={})".format(num_steps))

        for _ in range(num_steps):
            self.automaton.step()
            r, self._phi = compute_step_reward(self.automaton, self._phi, self.config)
            self.total_reward += r
            if is_xor_solved(self.automaton):
                self.evaluates_xor = True
                # Don't early-break — let the policy keep collecting per-step shaping
                # so it learns to *maintain* the solved state, not just touch it.

        return self.total_reward
