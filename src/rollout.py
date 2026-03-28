import json
from datetime import datetime
from typing import List, Tuple

from src.env import DummyTerminalBenchEnv
from src.metrics import compute_episode_metrics
from src.trajectory import Transition, Episode


def is_verbose_failure(normalized_action: str) -> bool:
    return len(normalized_action.split()) > 3 or len(normalized_action) > 20


def build_prompt(instruction: str, previous_attempts=None) -> str:
    previous_attempts = previous_attempts or []

    prompt = (
        "You are an agent solving a simple verifiable task.\n"
        "Return only the final answer text.\n\n"
        f"Task: {instruction}\n"
    )

    if previous_attempts:
        prompt += "\nPrevious attempts:\n"
        for i, attempt in enumerate(previous_attempts, start=1):
            prompt += f"{i}. {attempt}\n"

    prompt += "\nAnswer:"
    return prompt


def generate_action(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int,
    do_sample: bool,
    previous_attempts=None,
) -> str:
    import torch

    prompt = build_prompt(instruction, previous_attempts=previous_attempts)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        action = decoded.split("Answer:", 1)[1].strip()
    else:
        action = decoded.strip()

    action = action.splitlines()[0].strip()
    return action


def run_rollouts(model, tokenizer, env: DummyTerminalBenchEnv, tasks, config) -> Tuple[List[dict], float, float]:
    all_episodes = []
    total_reward = 0.0
    total_success = 0

    for task in tasks:
        obs = env.reset(task)
        done = False
        previous_attempts = []
        transitions = []
        final_info = None

        last_failed_normalized_action = None
        consecutive_same_failure_count = 0
        max_consecutive_same_failures = 3

        print(f"Task: {task.task_id}")
        print("  Instruction:", obs["instruction"])

        while not done:
            action = generate_action(
                model=model,
                tokenizer=tokenizer,
                instruction=obs["instruction"],
                max_new_tokens=config["generation"]["max_new_tokens"],
                do_sample=config["generation"]["do_sample"],
                previous_attempts=previous_attempts,
            )

            next_obs, reward, done, info = env.step(action)

            normalized_action = info["normalized_action"]

            if not info["success"]:
                if normalized_action == last_failed_normalized_action:
                    consecutive_same_failure_count += 1
                else:
                    last_failed_normalized_action = normalized_action
                    consecutive_same_failure_count = 1

                should_stop_early = (
                    info["task_type"] == "exact_match"
                    and is_verbose_failure(normalized_action)
                    and consecutive_same_failure_count >= max_consecutive_same_failures
                )

                if should_stop_early:
                    print(
                        "    Early stopping triggered after repeated consecutive failed attempt: "
                        f"{repr(normalized_action)}"
                    )
                    done = True
                    reward = 0.0
                    info = info.copy()
                    info["early_stop"] = True
            else:
                last_failed_normalized_action = None
                consecutive_same_failure_count = 0

            transitions.append(
                Transition(
                    observation=obs.copy(),
                    action=action,
                    reward=reward,
                    done=done,
                    info=info.copy(),
                )
            )

            previous_attempts.append(action)
            final_info = info
            obs = next_obs

            print(f"  Step {len(transitions)}:")
            print("    Raw action:", repr(action))
            print("    Normalized action:", repr(info["normalized_action"]))
            print("    Reward:", reward)
            print("    Done:", done)
            print("    Success:", info["success"])

        episode = Episode(task_id=task.task_id, transitions=transitions)
        all_episodes.append(episode.to_dict())

        total_reward += episode.total_reward
        total_success += int(episode.success)

        print("  Final expected:", repr(final_info["expected_answer"]))
        print("  Episode reward:", episode.total_reward)
        print("  Episode success:", episode.success)
        print("  Number of transitions:", len(episode.transitions))
        print()

    avg_reward = total_reward / len(tasks)
    success_rate = total_success / len(tasks)

    return all_episodes, avg_reward, success_rate


def save_rollouts(all_episodes, config, tasks, avg_reward: float, success_rate: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/rollouts_{timestamp}.json"
    metrics = compute_episode_metrics(all_episodes)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "num_tasks": len(tasks),
                "average_reward": avg_reward,
                "success_rate": success_rate,
                "success_rate_by_task_type": metrics["success_rate_by_task_type"],
                "average_reward_by_task_type": metrics["average_reward_by_task_type"],
                "average_steps_per_episode": metrics["average_steps_per_episode"],
                "average_steps_per_successful_episode": metrics["average_steps_per_successful_episode"],
                "episodes": all_episodes,
            },
            f,
            indent=2,
        )

    return output_path
