# RLVR Environment for TerminalBench

This repository implements an RLVR (Reinforcement Learning from Virtual Reward) environment designed to improve a language model's performance on TerminalBench-style verifiable tasks. The system uses TRL (Transformer Reinforcement Learning) as the RL framework with GRPO (Group Relative Policy Optimization) as the training algorithm, targeting Qwen2.5-0.5B-Instruct as the base model.

The goal is to make each design decision — state representation, action space, rewards, metrics — explicit so reviewers can understand how the components work together. Every component is wired end-to-end: the environment verifies model outputs, the reward functions score them, and the TRL GRPOTrainer uses those scores to update the policy.

---

## 1. Framework Selection: TRL

I selected **TRL (Transformer Reinforcement Learning)** by Hugging Face as the RL framework.

**Why TRL:**

- **Native HuggingFace integration.** TRL is built on top of `transformers` and `accelerate`, so it works directly with any HuggingFace model checkpoint without adapter code. Loading `Qwen2.5-0.5B-Instruct` and running GRPO requires minimal boilerplate.
- **First-class GRPO support.** TRL provides `GRPOTrainer` and `GRPOConfig` out of the box, which is the algorithm I chose for this project. Other frameworks require more infrastructure setup for the same capability.
- **Custom reward functions.** TRL's `GRPOTrainer` accepts arbitrary Python callables as reward functions, making it trivial to plug in domain-specific verification logic like the exact_match and arithmetic verifiers in this project.
- **Active maintenance.** TRL is actively maintained by Hugging Face with regular releases and a large user community, which makes debugging and extending the prototype straightforward.
- **Right-sized for prototyping.** Frameworks like OpenRLHF and verl are designed for distributed, production-scale RL training. For a prototype with a 0.5B parameter model and small task sets, TRL provides the right level of abstraction without unnecessary infrastructure complexity.

**Alternatives considered:**

- **OpenRLHF:** Strong distributed training support, but excessive infrastructure for a single-GPU prototype.
- **verl:** Good for large-scale RL, but the setup overhead is not justified at this scale.
- **Custom PPO/REINFORCE:** Would require implementing gradient estimation, KL penalties, and advantage computation from scratch — TRL already handles this correctly.

The training script (`src/train.py`) imports and uses TRL's `GRPOConfig` and `GRPOTrainer` directly, with two custom reward functions passed to the trainer.

---

## 2. Benchmark Selection: TerminalBench

I chose **TerminalBench** as the primary target benchmark. TerminalBench tasks are verifiable challenges where the model's output can be deterministically checked against an expected answer. This makes them ideal for RLVR because:

- **Deterministic verification.** Each task has a clear correct answer, eliminating the need for a learned reward model. The reward is "virtual" — computed by a verifier function rather than a human or neural judge.
- **Diverse task types.** The benchmark includes exact-match, arithmetic, and counting tasks, providing variety for the policy to learn across.
- **Clean reward signal.** Binary success/failure from verification produces a sparse but unambiguous reward signal that GRPO is designed to handle.

The evaluation dataset (`data/eval_terminalbench_tasks.json`) contains 5 tasks across three types. Training uses an expanded set: 6 manual tasks plus 20 synthetically generated tasks (26 total).

---

## 3. Environment Creation

### 3.1 State/Observation Space Definition

The observation space is a dictionary containing:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Unique identifier for the current task |
| `instruction` | string | The natural language task instruction |
| `step_count` | int | Number of steps taken so far in this episode |
| `task_type` | string | Category: `exact_match`, `arithmetic`, or `counting` |

Additionally, the prompt construction (`build_prompt()` in `src/rollout.py`) augments the observation with **previous attempts**, so the model can condition on its prior failed outputs:

```
You are an agent solving a simple verifiable task.
Return only the final answer text.

Task: {instruction}

Previous attempts:
1. {attempt_1}
2. {attempt_2}

Answer:
```

**Design rationale:** The observation focuses on what the policy needs to make a decision — the task description, its type (which determines verification logic), and a progress indicator. Including previous attempts enables the model to avoid repeating mistakes, which is important for exact_match tasks where the model might initially generate verbose explanations instead of the bare answer.

### 3.2 Action Space Definition

The action space is the **decoded text string** produced by the model's generation step. At each step:

1. The model receives the prompt (observation + attempt history).
2. It generates up to `max_new_tokens` (default: 16) tokens.
3. The generated text is decoded and the first line is extracted.
4. This raw text string is the action passed to the environment.

**Normalization:** Before verification, the action is normalized by `normalize_text()` which strips whitespace, removes common prefixes ("The final answer is", "Answer:", "Final answer:"), and strips quotes and punctuation.

**Design rationale:** Keeping actions as raw text strings (rather than token IDs or structured commands) means the policy learns through the same generation mechanism it uses at inference time. The normalization step ensures that stylistic differences don't cause false negatives in verification.

### 3.3 Reward Function Design (2 Reward Functions)

The project implements **two distinct reward functions**, both compatible with TRL's `GRPOTrainer` interface:

#### Reward Function 1: Accuracy Reward (`accuracy_reward_fn`)

```
R_accuracy(completion) = 1.0   if verify_action(completion, expected) == True
                         0.0   otherwise
```

This is the primary RLVR signal. It uses the environment's verification logic (`verify_action` in `src/env.py`) to check whether the model's output is correct:

- **exact_match:** Raw string comparison against the expected answer.
- **arithmetic:** Normalized string comparison (strips formatting).
- **counting:** Parses the instruction, counts letter occurrences, and compares to the model's numeric output.

**Why this reward matters:** This is the core virtual reward — derived from deterministic verification rather than human preference. It directly incentivizes the model to produce correct answers.

#### Reward Function 2: Format Bonus Reward (`format_bonus_reward_fn`)

```
R_format(completion) = conciseness_score + type_format_score

conciseness_score = 0.3   if word_count <= 1
                   0.1   if word_count <= 3
                   0.0   otherwise

type_format_score = 0.2   if (arithmetic/counting AND output is numeric)
                   0.2   if (exact_match AND 0 < length < 50)
                   0.0   otherwise
```

This reward provides **partial credit for format compliance** even when the answer is incorrect:

- **Conciseness:** Shorter answers score higher, discouraging verbose explanations.
- **Type-appropriate format:** Numeric outputs for arithmetic/counting, reasonable-length outputs for exact_match.

**Why this reward matters:** The accuracy reward is sparse — the model gets 0.0 for any wrong answer regardless of how close it was. The format bonus provides a denser learning signal during early training when the model rarely produces exact correct answers. It shapes the model toward the right *kind* of output (short, correctly formatted) before it learns to produce the right *content*. This is a form of reward shaping that accelerates learning without misaligning the objective.

Both reward function classes are also defined in `src/reward.py` (`RewardFunction` and `FormatBonusRewardFunction`) for use in the rollout evaluation pipeline.

#### Combined Reward in GRPO

TRL's `GRPOTrainer` accepts multiple reward functions and combines them. The total reward per completion is:

```
R_total = R_accuracy + R_format
```

This gives a maximum reward of 1.5 (correct + concise + right format) and a minimum of 0.0. The group-relative normalization in GRPO ensures that the absolute scale doesn't matter — what matters is the relative ranking of completions within each group.

#### Step Penalty (Rollout Evaluation)

During rollout evaluation (`src/eval.py`), the `RewardFunction` class also applies a step penalty of `-0.01` per step:

```
R_episode = success_reward - (step_count * step_penalty)
```

This encourages the policy to solve tasks in fewer steps during multi-step evaluation episodes.

---

## 4. Model Selection and Training Plan

### 4.1 Base Model

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`

**Justification:**

- **Instruction-tuned.** The model already understands instruction-following, giving GRPO a strong starting point rather than training from a raw base model.
- **Small enough for prototyping.** At 0.5B parameters, the model trains quickly on a single GPU, enabling rapid iteration on reward design and hyperparameters.
- **Capable enough for the tasks.** Despite its size, Qwen2.5-0.5B-Instruct handles simple text generation tasks like exact matching and basic arithmetic, so the initial success rate is non-zero — critical for GRPO to get a learning signal.
- **HuggingFace ecosystem.** Available directly through `transformers`, compatible with TRL without custom loading code.

### 4.2 RL Algorithm: GRPO

**Algorithm:** Group Relative Policy Optimization (GRPO)

**Why GRPO:**

- **No critic network needed.** Unlike PPO, GRPO estimates advantages by comparing completions within a group (multiple generations per prompt), eliminating the need to train a separate value network. This simplifies the pipeline and reduces compute.
- **Well-suited for sparse rewards.** GRPO normalizes rewards within each group of completions, so even when most completions score 0.0, the relative differences still produce meaningful gradients.
- **Stable with deterministic verification.** Because TerminalBench verification is deterministic, the reward signal has zero noise. GRPO's group-relative normalization handles this cleanly.
- **TRL provides a production-quality implementation.** `GRPOTrainer` handles generation, reward computation, advantage estimation, and policy updates in a single training loop.

### 4.3 Dataset Configuration

| Parameter | Value |
|---|---|
| Manual tasks | 6 (from `data/dummy_terminalbench_tasks.json`) |
| Synthetic tasks | 20 (generated by `src/synthetic_tasks.py`) |
| Total training prompts | 26 |
| Eval tasks | 5 (from `data/eval_terminalbench_tasks.json`) |
| Task type distribution | ~33% exact_match, ~33% arithmetic, ~33% counting |
| Generations per prompt | 4 (GRPO group size) |

The synthetic task generator (`src/synthetic_tasks.py`) programmatically creates tasks across all three types:

- **Counting:** Random letter/word combinations from a vocabulary of 20 words.
- **Arithmetic:** Random addition, subtraction, and multiplication with varied operand ranges.
- **Exact match:** Random target strings from a vocabulary of 15 common words.

Using a fixed seed (`seed=42`) ensures reproducibility across runs.

### 4.4 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | `1e-5` | Conservative LR to avoid catastrophic forgetting of instruction-following capability |
| `per_device_train_batch_size` | `2` | Small batch for single-GPU training; GRPO groups provide effective larger batch |
| `max_steps` | `100` | Sufficient for convergence on 26 prompts with 4 generations each |
| `max_completion_length` | `16` | Tasks require short answers; limits compute waste on verbose generations |
| `num_generations` | `4` | GRPO group size — 4 completions per prompt for relative ranking |
| `optimizer` | AdamW (TRL default) | Standard choice for transformer fine-tuning |
| `logging_steps` | `10` | Frequent enough to monitor training dynamics |
| `save_steps` | `50` | Checkpoint at midpoint and end |

### 4.5 Evaluation Metrics

The following metrics are computed by `src/metrics.py` and reported after each evaluation run:

| Metric | Description |
|---|---|
| **Average reward per episode** | Mean total reward across all episodes; reflects both accuracy and efficiency |
| **Benchmark accuracy (success rate)** | Fraction of tasks solved correctly; the primary performance metric |
| **Average steps per episode** | Mean number of steps before episode termination; lower is better |
| **Average steps per successful episode** | Mean steps for tasks that were solved; measures efficiency of correct solutions |
| **Per-task-type success rate** | Success rate broken down by exact_match, arithmetic, counting |
| **Per-task-type average reward** | Average reward broken down by task type; identifies weak areas |

These metrics are printed to stdout and persisted in the rollout JSON files (`outputs/rollouts_*.json`) alongside the full episode traces.

---

## 5. Stretch Goals

### 5.1 Synthetic Task Generation

`src/synthetic_tasks.py` provides a programmatic task generator that creates novel tasks conforming to the TerminalBench format:

- `generate_counting_task()` — Random letter/word counting challenges.
- `generate_arithmetic_task()` — Random arithmetic (addition, subtraction, multiplication).
- `generate_exact_match_task()` — Random string echo tasks.
- `generate_task_set(n, seed)` — Generates a mixed batch of n tasks with deterministic seeding.

This expands the effective training set beyond the 6 hand-written tasks and enables curriculum experiments with larger or more varied task distributions. The training script automatically generates 20 synthetic tasks and combines them with the manual tasks.

### 5.2 Dual Environment Creation

The repository includes environments for **both** TerminalBench and tau2 bench:

**TerminalBench (primary):**

- `src/env.py` — `DummyTerminalBenchEnv` for synthetic verifiable tasks with full verification logic.
- `src/terminalbench_env.py` — `TerminalBenchEnv` scaffold for real TerminalBench2 / Harbor-backed tasks.
- `src/terminalbench_backend.py` — Backend interface with `DummyTerminalBenchBackend` stub for future Harbor integration.
- `src/terminalbench_adapter.py` — Task loader for TerminalBench2 task format.
- `src/terminalbench_demo.py` — Exercises the TerminalBench2 scaffold stack.

**tau2 bench (prototype):**

- `src/tau2_env.py` — `Tau2Env` for conversational tool-use tasks.
- Multi-turn conversation history as observation space.
- Tool call actions with simulated execution.
- Completion-based and tool-accuracy rewards.

Both TerminalBench environments share the `BaseEnv` interface (`src/base_env.py`), making them interchangeable in the rollout pipeline.

### 5.3 Scaling Considerations

**Scaling down (smaller models, e.g., <1B params):**

- Use simpler tasks: more exact_match, fewer arithmetic/counting.
- Increase `success_reward` relative to `step_penalty` (sharper signal).
- Reduce `max_episode_steps` (shorter episodes).
- Smaller `num_generations` in GRPO (2 instead of 4).
- Smaller batch size.
- Start curriculum with only exact_match tasks, add arithmetic/counting after warm-up.

**Scaling up (larger models, e.g., 7B+):**

- Add harder tasks: multi-step arithmetic, longer exact-match strings, complex counting.
- Increase `max_episode_steps` to allow more complex reasoning chains.
- Larger `num_generations` (8-16) for better GRPO advantage estimates.
- Larger batch size with gradient accumulation.
- Lower learning rate (5e-6 or lower) for stability.
- Interleave all task types from the start — larger models handle variety sooner.
- Consider LoRA/QLoRA for memory efficiency at 7B+ scale.

---

## Project Structure

```
├── configs/
│   └── base.yaml                    # Training and environment configuration
├── data/
│   ├── dummy_terminalbench_tasks.json   # Training tasks (6 manual)
│   ├── eval_terminalbench_tasks.json    # Evaluation tasks (5)
│   └── terminalbench_stub_tasks.json    # TerminalBench2 stub tasks
├── outputs/                         # Rollout logs and trained models
├── src/
│   ├── __init__.py
│   ├── base_env.py                  # Abstract base environment interface
│   ├── env.py                       # DummyTerminalBenchEnv (synthetic tasks)
│   ├── eval.py                      # Evaluation script (rollout + metrics)
│   ├── metrics.py                   # Episode metrics computation
│   ├── reward.py                    # Reward functions (RewardFunction + FormatBonusRewardFunction)
│   ├── rollout.py                   # Rollout execution and logging
│   ├── synthetic_tasks.py           # Programmatic task generation
│   ├── tasks.py                     # Task loading utilities
│   ├── tau2_env.py                  # tau2 bench environment (stretch goal)
│   ├── terminalbench_adapter.py     # TerminalBench2 task format adapter
│   ├── terminalbench_backend.py     # TerminalBench2 backend interface
│   ├── terminalbench_demo.py        # TerminalBench2 scaffold demo
│   ├── terminalbench_env.py         # TerminalBench2 environment scaffold
│   ├── train.py                     # TRL GRPO training script
│   └── trajectory.py               # Transition and Episode data structures
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run GRPO training (uses TRL GRPOTrainer with both reward functions)
python -m src.train

# Run evaluation (rollout-based with metrics)
python -m src.eval

# Demo TerminalBench2 scaffold
python -m src.terminalbench_demo
```

## Current Limitations

- The synthetic environment covers three task types (exact_match, arithmetic, counting); extending to file-system or command-line tasks requires a real TerminalBench2 backend.
- GRPO training on 26 tasks is a proof-of-concept; production use would need hundreds or thousands of diverse tasks.
- No distributed training support — the prototype targets single-GPU training.
- Rollout logging emits one JSON per run with no automated comparison or visualization tooling.
