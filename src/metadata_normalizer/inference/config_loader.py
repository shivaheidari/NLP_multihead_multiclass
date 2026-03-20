"""
Configuration loader for the NLP multi-head multi-class classification project.

This module provides functionality to load the project's main YAML configuration
file, which contains settings for data processing, model architecture, training,
and inference.
"""
from pathlib import Path
import yaml

def load_project_config(config_rel_path: str = "configs/config.yaml") -> dict:
    """
    Load the project configuration from a YAML file.

    Args:
        config_rel_path (str): The relative path to the configuration file from
            the project root directory. Defaults to "configs/config.yaml".

    Returns:
        dict: A dictionary containing the parsed configuration settings.
    """
    root = Path(__file__).resolve().parents[3]
    cfg_path = root / config_rel_path
    with open(cfg_path) as f:
        return yaml.safe_load(f)