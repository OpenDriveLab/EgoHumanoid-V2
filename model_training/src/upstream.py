"""Load entry points from the installed, pinned OpenPI checkout."""

import importlib.util
import sys
from pathlib import Path

import openpi.training.config


def load_script(name):
    root = Path(openpi.training.config.__file__).resolve().parents[3]
    path = root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"eh2_openpi_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
