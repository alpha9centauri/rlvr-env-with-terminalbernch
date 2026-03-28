from pathlib import Path
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.env import DummyTerminalBenchEnv
from src.reward import RewardFunction
from src.tasks import load_tasks
from src.rollout import run_rollouts, save_rollouts


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

    tasks = load_tasks()

    model_name = config["model"]["name"]
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Reward config:", reward_fn.describe())
    print(f"Loaded {len(tasks)} tasks\n")

    all_episodes, avg_reward, success_rate = run_rollouts(
        model=model,
        tokenizer=tokenizer,
        env=env,
        tasks=tasks,
        config=config,
    )

    print("Summary")
    print("-------")
    print("Average reward:", avg_reward)
    print("Success rate:", success_rate)

    output_path = save_rollouts(
        all_episodes=all_episodes,
        config=config,
        tasks=tasks,
        avg_reward=avg_reward,
        success_rate=success_rate,
    )

    print(f"Saved rollouts to: {output_path}")