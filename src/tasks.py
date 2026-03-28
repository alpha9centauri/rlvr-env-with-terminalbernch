import json
from pathlib import Path
from typing import List

from src.env import Task
from src.terminalbench_adapter import TerminalBenchTask


def load_tasks(task_path: str = "data/dummy_terminalbench_tasks.json") -> List[Task]:
    path = Path(task_path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    with path.open("r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    tasks = []
    for item in raw_tasks:
        tasks.append(
            Task(
                task_id=item["task_id"],
                instruction=item["instruction"],
                expected_answer=item["expected_answer"],
                task_type=item["task_type"],
            )
        )

    return tasks


def load_terminalbench_tasks(
    task_path: str = "data/terminalbench_stub_tasks.json",
) -> List[TerminalBenchTask]:
    from src.terminalbench_adapter import load_terminalbench_tasks

    return load_terminalbench_tasks(task_path=task_path)
