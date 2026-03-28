from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class Tau2Task:
    task_id: str
    user_prompt: str
    expected_tool_call: Optional[str] = None
    completion_keyword: str = "done"


class Tau2Env:
    """
    Prototype tau2-style environment for conversational tool use tasks.
    This is intentionally lightweight and not a full tau2 integration.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.current_task: Optional[Tau2Task] = None
        self.turn_count = 0
        self.conversation_history: List[Dict[str, str]] = []
        self.tool_outputs: List[str] = []
        self.done = False

    def reset(self, task: Tau2Task) -> Dict[str, Any]:
        self.current_task = task
        self.turn_count = 0
        self.done = False
        self.conversation_history = [{"role": "user", "text": task.user_prompt}]
        self.tool_outputs = []
        return self._build_observation()

    def step(self, action: Dict[str, Optional[str]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.current_task is None:
            raise ValueError("Must reset the environment with a task before stepping.")
        if self.done:
            raise ValueError("Episode already finished; call reset().")

        self.turn_count += 1
        assistant_response = action.get("assistant_response", "").strip()
        optional_tool_call = action.get("optional_tool_call")

        if assistant_response:
            self.conversation_history.append({"role": "assistant", "text": assistant_response})

        tool_output: Optional[str] = None
        if optional_tool_call:
            tool_output = f"Simulated output for {optional_tool_call}"
            self.tool_outputs.append(tool_output)

        reward = self._compute_reward(assistant_response, optional_tool_call)
        self.done = self.done or self._check_completion(assistant_response) or self.turn_count >= self.max_turns

        info = {
            "tool_call": optional_tool_call,
            "tool_output": tool_output,
        }

        return self._build_observation(), reward, self.done, info

    def _compute_reward(self, assistant_response: str, tool_call: Optional[str]) -> float:
        reward = 0.0

        if self.current_task and self.current_task.expected_tool_call and tool_call == self.current_task.expected_tool_call:
            reward += 0.5

        if self._check_completion(assistant_response):
            reward += 1.0

        return reward

    def _check_completion(self, assistant_response: str) -> bool:
        if not self.current_task:
            return False
        keyword = self.current_task.completion_keyword.lower()
        return keyword in assistant_response.lower()

    def _build_observation(self) -> Dict[str, Any]:
        return {
            "conversation_history": list(self.conversation_history),
            "tool_outputs": list(self.tool_outputs),
            "turn_count": self.turn_count,
        }
