from src.reward import RewardFunction
from src.terminalbench_adapter import load_terminalbench_tasks
from src.terminalbench_backend import DummyTerminalBenchBackend
from src.terminalbench_env import TerminalBenchEnv


def main():
    tasks = load_terminalbench_tasks()
    reward_fn = RewardFunction(success_reward=1.0, failure_reward=0.0, step_penalty=0.01)
    env = TerminalBenchEnv(reward_fn=reward_fn, backend=DummyTerminalBenchBackend(), max_episode_steps=5)

    for task in tasks[:2]:
        obs = env.reset(task)
        print("Reset observation:", obs)

        placeholder_action = {"command": "noop", "args": ["placeholder"]}
        observation, reward, done, info = env.step(placeholder_action)
        print("Step result:", {"observation": observation, "reward": reward, "done": done, "info": info})

        if done:
            break


if __name__ == "__main__":
    main()
