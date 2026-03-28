from typing import Dict


class RewardFunction:
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