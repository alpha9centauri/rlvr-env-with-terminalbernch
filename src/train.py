from pathlib import Path
import yaml


def load_config(config_path: str = "configs/base.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    config = load_config()
    print("Loaded config:")
    print(config)