from pathlib import Path

import yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from src.env import normalize_text, verify_action
from src.tasks import load_tasks
from src.rollout import build_prompt
from src.synthetic_tasks import generate_task_set


def load_config(config_path: str = "configs/base.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_grpo_dataset(tasks) -> Dataset:
    """Convert tasks into a HuggingFace Dataset with prompt column for GRPOTrainer."""
    records = {"prompt": [], "expected_answer": [], "task_type": [], "instruction": []}
    for task in tasks:
        prompt = build_prompt(task.instruction)
        records["prompt"].append(prompt)
        records["expected_answer"].append(task.expected_answer)
        records["task_type"].append(task.task_type)
        records["instruction"].append(task.instruction)
    return Dataset.from_dict(records)


def accuracy_reward_fn(completions, expected_answer, task_type, instruction, **kwargs):
    """TRL-compatible reward function: +1.0 for correct verification, 0.0 otherwise."""
    rewards = []
    for completion, expected, ttype, instr in zip(completions, expected_answer, task_type, instruction):
        text = completion.strip() if isinstance(completion, str) else str(completion)
        normalized = normalize_text(text)
        try:
            success = verify_action(normalized, expected, ttype, instr)
        except ValueError:
            success = False
        rewards.append(1.0 if success else 0.0)
    return rewards


def format_bonus_reward_fn(completions, expected_answer, task_type, **kwargs):
    """TRL-compatible reward function: partial credit for output format compliance."""
    rewards = []
    for completion, expected, ttype in zip(completions, expected_answer, task_type):
        text = completion.strip() if isinstance(completion, str) else str(completion)
        normalized = normalize_text(text)
        score = 0.0

        word_count = len(normalized.split())
        if word_count <= 1:
            score += 0.3
        elif word_count <= 3:
            score += 0.1

        if ttype in ("arithmetic", "counting"):
            if normalized.isdigit():
                score += 0.2
        elif ttype == "exact_match":
            if 0 < len(normalized) < 50:
                score += 0.2

        rewards.append(score)
    return rewards


if __name__ == "__main__":
    config = load_config()

    model_name = config["model"]["name"]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = load_tasks("data/dummy_terminalbench_tasks.json")
    synthetic_tasks = generate_task_set(n=20)
    all_tasks = tasks + synthetic_tasks

    print(f"Loaded {len(tasks)} manual tasks + {len(synthetic_tasks)} synthetic tasks = {len(all_tasks)} total")

    dataset = build_grpo_dataset(all_tasks)

    grpo_config = GRPOConfig(
        output_dir="outputs/grpo_training",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        max_steps=config["training"]["max_steps"],
        learning_rate=config["training"]["learning_rate"],
        max_completion_length=config["generation"]["max_new_tokens"],
        logging_steps=10,
        save_steps=50,
        num_generations=4,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward_fn, format_bonus_reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Saving trained model...")
    trainer.save_model("outputs/grpo_model")
    tokenizer.save_pretrained("outputs/grpo_model")

    print("Training complete.")
