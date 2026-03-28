import re
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from src.base_env import BaseEnv
from src.reward import RewardFunction


@dataclass
class Task:
    task_id: str
    instruction: str
    expected_answer: str
    task_type: str


def normalize_text(text: str) -> str:
    text = text.strip()

    prefixes = [
        "The final answer is",
        "Final answer:",
        "Answer:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text.strip("\"'`.,:;!? ")


def verify_action(action: str, expected_answer: str, task_type: str, instruction: str) -> bool:
    raw_action = action.strip()
    raw_expected = expected_answer.strip()

    norm_action = normalize_text(action)
    norm_expected = normalize_text(expected_answer)

    if task_type == "exact_match":
        return raw_action == raw_expected

    if task_type == "arithmetic":
        return norm_action == norm_expected

    if task_type == "counting":
        return compute_counting_answer(instruction) == raw_action

    raise ValueError(f"Unknown task_type: {task_type}")


def compute_counting_answer(instruction: str) -> str:
    letter_match = re.search(r"letter '(.{1})'", instruction)
    word_match = re.search(r"word '([^']+)'", instruction)

    if not letter_match or not word_match:
        raise ValueError("Counting instruction must specify letter and word")

    letter_char = letter_match.group(1).lower()
    word_text = word_match.group(1).lower()
    count = word_text.count(letter_char)
    return str(count)


class DummyTerminalBenchEnv(BaseEnv):
    def __init__(self, reward_fn: RewardFunction, max_episode_steps: int = 20):
        self.reward_fn = reward_fn
        self.max_episode_steps = max_episode_steps
        self.current_task = None
        self.step_count = 0
        self.done = False

    def reset(self, task: Task) -> Dict[str, Any]:
        self.current_task = task
        self.step_count = 0
        self.done = False

        return {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "step_count": self.step_count,
            "task_type": task.task_type,
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.current_task is None:
            raise ValueError("Environment must be reset with a task before calling step().")

        if self.done:
            raise ValueError("Episode is already done. Call reset() to start a new episode.")

        self.step_count += 1

        normalized_action = normalize_text(action)
        normalized_expected = normalize_text(self.current_task.expected_answer)

        success = verify_action(
            action=action,
            expected_answer=self.current_task.expected_answer,
            task_type=self.current_task.task_type,
            instruction=self.current_task.instruction,
        )

        reached_limit = self.step_count >= self.max_episode_steps
        self.done = success or reached_limit

        reward = self.reward_fn.compute(
            success=success,
            step_count=self.step_count,
            done=self.done,
        )

        observation = {
            "task_id": self.current_task.task_id,
            "instruction": self.current_task.instruction,
            "step_count": self.step_count,
            "task_type": self.current_task.task_type,
        }

        info = {
            "success": success,
            "task_type": self.current_task.task_type,
            "expected_answer": self.current_task.expected_answer,
            "normalized_expected_answer": normalized_expected,
            "agent_action": action,
            "normalized_action": normalized_action,
        }

        return observation, reward, self.done, info
