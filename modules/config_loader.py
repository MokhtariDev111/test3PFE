"""
config_loader.py
----------------
Utility to load config.yaml from the project root.
Every module imports this to access settings centrally.
"""

import yaml
from pathlib import Path

# Resolve the project root (one level up from this file if inside modules/,
# or the same directory if running from the root).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if _HERE.name == "modules" else _HERE
CONFIG_PATH = _ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load and return the YAML configuration as a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# Singleton — loaded once and reused across imports
CONFIG: dict = load_config()


if __name__ == "__main__":
    import json
    print(json.dumps(CONFIG, indent=2, default=str))
