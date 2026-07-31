from pathlib import Path

from config_loader import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dpo.json"


def load_config() -> Config:
    return Config(CONFIG_PATH)
