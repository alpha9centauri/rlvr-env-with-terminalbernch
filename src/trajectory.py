from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class Transition:
    observation: Dict[str, Any]
    action: str
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class Episode:
    task_id: str
    transitions: List[Transition]

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)

    @property
    def success(self) -> bool:
        if not self.transitions:
            return False
        return bool(self.transitions[-1].info.get("success", False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_reward": self.total_reward,
            "success": self.success,
            "transitions": [asdict(t) for t in self.transitions],
        }