from typing import Any, Dict, List


def compute_episode_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    type_reward_sums: Dict[str, float] = {}
    type_success_counts: Dict[str, int] = {}
    total_steps = 0
    successful_steps = 0
    successful_episodes = 0

    for episode in episodes:
        transitions = episode.get("transitions", [])
        steps = len(transitions)
        total_steps += steps

        if transitions:
            task_type = transitions[0]["observation"].get("task_type", "unknown")
        else:
            task_type = "unknown"

        type_counts[task_type] = type_counts.get(task_type, 0) + 1
        type_reward_sums[task_type] = type_reward_sums.get(task_type, 0.0) + episode.get("total_reward", 0.0)

        if episode.get("success"):
            type_success_counts[task_type] = type_success_counts.get(task_type, 0) + 1
            successful_steps += steps
            successful_episodes += 1

    success_rate_by_task_type = {
        task_type: type_success_counts.get(task_type, 0) / count
        for task_type, count in type_counts.items()
        if count > 0
    }
    average_reward_by_task_type = {
        task_type: type_reward_sums.get(task_type, 0.0) / count
        for task_type, count in type_counts.items()
        if count > 0
    }

    average_steps_per_episode = total_steps / len(episodes) if episodes else 0.0
    average_steps_per_successful_episode = (
        successful_steps / successful_episodes if successful_episodes else 0.0
    )

    return {
        "success_rate_by_task_type": success_rate_by_task_type,
        "average_reward_by_task_type": average_reward_by_task_type,
        "average_steps_per_episode": average_steps_per_episode,
        "average_steps_per_successful_episode": average_steps_per_successful_episode,
    }
