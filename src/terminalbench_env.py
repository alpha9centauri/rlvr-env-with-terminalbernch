from typing import Any, Dict, Optional

from src.base_env import BaseEnv
from src.reward import RewardFunction
from src.terminalbench_backend import DummyTerminalBenchBackend, TerminalBenchBackend
from src.terminalbench_adapter import TerminalBenchTask


class TerminalBenchEnv(BaseEnv):
    """
    Scaffold environment for future TerminalBench2 / Harbor-backed tasks.
    This is a lightweight adapter that keeps the same interface as DummyTerminalBenchEnv.
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        backend: Optional[TerminalBenchBackend] = None,
        max_episode_steps: int = 50,
    ):
        self.reward_fn = reward_fn
        self.backend = backend or DummyTerminalBenchBackend()
        self.max_episode_steps = max_episode_steps
        self.current_task: Optional[TerminalBenchTask] = None
        self.step_count = 0
        self.done = False

    def reset(self, task: TerminalBenchTask) -> Dict[str, Any]:
        self.current_task = task
        self.step_count = 0
        self.done = False
        self.backend.initialize_task(task)
        return self._build_observation()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_task is None:
            raise ValueError("Call reset() before stepping.")

        self.step_count += 1
        backend_response = self.backend.execute_action(self.current_task, action)
        verification = self.backend.verify_task(self.current_task)
        success = bool(verification.get("success", False))
        reached_limit = self.step_count >= min(self.max_episode_steps, self.current_task.max_episode_steps)
        self.done = self.done or success or reached_limit

        reward = self.reward_fn.compute(success, self.step_count, self.done)

        info = {
            "success": success,
            "backend_status": backend_response.get("status"),
            "verifier_type": self.current_task.verifier_type,
            "agent_action": action,
            "verification": verification,
            "task_type": "terminalbench2",
            "normalized_action": str(action),
        }

        observation = self._build_observation()
        return observation, reward, self.done, info

    def _build_observation(self) -> Dict[str, Any]:
        if self.current_task is None:
            return {}
        return {
            "task_id": self.current_task.task_id,
            "instruction": self.current_task.instruction,
            "benchmark_name": self.current_task.benchmark_name,
            "step_count": self.step_count,
            "metadata": self.current_task.metadata,
        }
