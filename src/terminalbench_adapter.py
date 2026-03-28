import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TerminalBenchTask:
    task_id: str
    instruction: str
    benchmark_name: str
    verifier_type: str
    max_episode_steps: int
    metadata: Dict[str, Any]


def load_terminalbench_tasks(task_path: str = "data/terminalbench_stub_tasks.json") -> List[TerminalBenchTask]:
    path = Path(task_path)
    if not path.exists():
        raise FileNotFoundError(f"TerminalBench task file not found: {task_path}")

    with path.open("r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    tasks = []
    for item in raw_tasks:
        tasks.append(
            TerminalBenchTask(
                task_id=item["task_id"],
                instruction=item["instruction"],
                benchmark_name=item["benchmark_name"],
                verifier_type=item["verifier_type"],
                max_episode_steps=item.get("max_episode_steps", 20),
                metadata=item.get("metadata", {}),
            )
        )

    return tasks
