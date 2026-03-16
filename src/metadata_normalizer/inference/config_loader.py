from pathlib import Path
import yaml

def load_project_config(config_rel_path: str = "configs/config.yaml") -> dict:
    root = Path(__file__).resolve().parents[3]
    cfg_path = root / config_rel_path
    with open(cfg_path) as f:
        return yaml.safe_load(f)