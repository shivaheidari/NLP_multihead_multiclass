from pathlib import Path
import yaml

def load_project_config(config_rel_path: str = "config/cofing.yaml") -> dict:
    root = Path(__file__).resolve().parent[2]
    cfg_path = root / config_rel_path
    with open(cfg_path) as f:
        return yaml.safe_load(f)