from typing import Dict


class RewardFunction:
    """Primary reward: accuracy with step penalty for efficiency."""

    def __init__(self, success_reward: float, failure_reward: float, step_penalty: float):
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.step_penalty = step_penalty

    def compute(self, success: bool, step_count: int, done: bool) -> float:
        if success:
            return self.success_reward - (step_count * self.step_penalty)

        if done:
            return self.failure_reward

        return -self.step_penalty

    def describe(self) -> Dict[str, float]:
        return {
            "success_reward": self.success_reward,
            "failure_reward": self.failure_reward,
            "step_penalty": self.step_penalty,
        }


class FormatBonusRewardFunction:
    """Secondary reward: partial credit for output format compliance.

    Provides a denser learning signal than pure accuracy by rewarding:
    - Concise outputs (single-word or short-phrase answers)
    - Type-appropriate format (numeric for arithmetic/counting, reasonable length for exact_match)
    """

    def __init__(self, conciseness_bonus: float = 0.3, format_bonus: float = 0.2):
        self.conciseness_bonus = conciseness_bonus
        self.format_bonus = format_bonus

    def compute(self, normalized_action: str, task_type: str) -> float:
        score = 0.0
        word_count = len(normalized_action.split())

        if word_count <= 1:
            score += self.conciseness_bonus
        elif word_count <= 3:
            score += self.conciseness_bonus * 0.33

        if task_type in ("arithmetic", "counting"):
            if normalized_action.isdigit():
                score += self.format_bonus
        elif task_type == "exact_match":
            if 0 < len(normalized_action) < 50:
                score += self.format_bonus

        return score

    def describe(self) -> Dict[str, float]:
        return {
            "conciseness_bonus": self.conciseness_bonus,
            "format_bonus": self.format_bonus,
        }
