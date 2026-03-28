from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseEnv(ABC):
    """
    Minimal environment interface shared by synthetic and TerminalBench2 adapters.
    """

    @abstractmethod
    def reset(self, task: Any) -> Dict[str, Any]:
        ...

    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        ...
