from pathlib import Path
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.env import DummyTerminalBenchEnv
from src.reward import RewardFunction
from src.tasks import load_tasks
from src.rollout import run_rollouts
from src.metrics import compute_episode_metrics


def load_config(config_path: str = "configs/base.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()

    reward_fn = RewardFunction(
        success_reward=config["reward"]["success_reward"],
        failure_reward=config["reward"]["failure_reward"],
        step_penalty=config["reward"]["step_penalty"],
    )

    env = DummyTerminalBenchEnv(
        reward_fn=reward_fn,
        max_episode_steps=config["environment"]["max_episode_steps"],
    )

    tasks = load_tasks("data/eval_terminalbench_tasks.json")

    model_name = config["model"]["name"]
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Running evaluation...")
    print(f"Loaded {len(tasks)} eval tasks\n")

    episodes, avg_reward, success_rate = run_rollouts(
        model=model,
        tokenizer=tokenizer,
        env=env,
        tasks=tasks,
        config=config,
    )

    metrics = compute_episode_metrics(episodes)

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Overall success rate: {success_rate:.2f}")
    print(f"Overall average reward: {avg_reward:.2f}")
    print(f"Average steps per episode: {metrics['average_steps_per_episode']:.2f}")
    print(f"Average steps per successful episode: {metrics['average_steps_per_successful_episode']:.2f}")

    print("\nPer-task-type success rate:")
    for task_type in sorted(metrics["success_rate_by_task_type"]):
        rate = metrics["success_rate_by_task_type"][task_type]
        print(f"  {task_type}: {rate:.2f}")

    print("\nPer-task-type average reward:")
    for task_type in sorted(metrics["average_reward_by_task_type"]):
        reward_avg = metrics["average_reward_by_task_type"][task_type]
        print(f"  {task_type}: {reward_avg:.2f}")
