from typing import Any, Dict, Optional

class TerminalBenchBackend:
    """
    Backend interface for running TerminalBench2 tasks.
    """

    def initialize_task(self, task: Any) -> None:
        raise NotImplementedError

    def execute_action(self, task: Any, action: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def verify_task(self, task: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def get_observation(self, task: Any) -> Dict[str, Any]:
        raise NotImplementedError


class DummyTerminalBenchBackend(TerminalBenchBackend):
    """
    Placeholder backend that returns canned responses for demo purposes.
    """

    def initialize_task(self, task: Any) -> None:
        task.metadata.setdefault("backend_state", {})

    def execute_action(self, task: Any, action: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "action_recorded", "tool_outputs": [], "last_action": action}

    def verify_task(self, task: Any) -> Dict[str, Any]:
        return {"success": False, "reason": "verification stub"}

    def get_observation(self, task: Any) -> Dict[str, Any]:
        return {"status": "ready", "benchmark": task.benchmark_name}
