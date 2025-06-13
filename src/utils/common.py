import os
import yaml
import json
from pathlib import Path
from typing import Any

def read_yaml(path_to_yaml: Path) -> dict:
    """Read yaml file and returns dictionary"""
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def create_dirs(path_to_directories: list, verbose=True):
    """Create list of directories"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Created directory at: {path}")

def save_json(path: Path, data: dict):
    """Save json data"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path: Path) -> dict:
    """Load json file"""
    with open(path) as f:
        content = json.load(f)
    return content