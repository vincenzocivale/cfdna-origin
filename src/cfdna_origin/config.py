"""YAML configuration: path resolution and experiment composition.

Paths come from `configs/paths.yaml` (committed, relative defaults) overlaid by `configs/paths.local.yaml`
(gitignored, machine-specific). Values may reference environment variables as `${VAR}` or `${VAR:-default}` and
other path keys as `{key}`. Relative paths resolve against the repository root.

An experiment YAML names a dataset, a model and a list of representations by file stem; `load_experiment` inlines
them so a run's `config.yaml` is fully self-contained.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def read_yaml(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand_env(value: str) -> str:
    def repl(m: re.Match) -> str:
        if m.group(1) in os.environ:
            return os.environ[m.group(1)]
        if m.group(2) is not None:
            return m.group(2)
        raise KeyError(f"environment variable {m.group(1)} is not set and has no default")

    return _ENV.sub(repl, value)


def load_paths(config_dir: Path = CONFIG_DIR) -> dict[str, Path]:
    raw = read_yaml(config_dir / "paths.yaml")
    local = config_dir / "paths.local.yaml"
    if local.exists():
        raw = deep_merge(raw, read_yaml(local))
    resolved: dict[str, Path] = {}
    pending = {k: _expand_env(str(v)) for k, v in raw.items()}
    for _ in range(len(pending) + 1):  # resolve {key} references in dependency order
        for key, value in list(pending.items()):
            refs = re.findall(r"\{([a-z_]+)\}", value)
            if all(r in resolved for r in refs):
                for r in refs:
                    value = value.replace("{" + r + "}", str(resolved[r]))
                p = Path(value).expanduser()
                resolved[key] = p if p.is_absolute() else (REPO_ROOT / p)
                del pending[key]
    if pending:
        raise ValueError(f"unresolvable path references: {pending}")
    return resolved


def resolve_path(value: str | Path, paths: dict[str, Path]) -> Path:
    """Resolve a config path value that may contain `{key}` path references and `${ENV}` variables."""
    value = _expand_env(str(value))
    for key, p in paths.items():
        value = value.replace("{" + key + "}", str(p))
    if "{" in value:
        raise KeyError(f"unknown path reference in {value!r}; known keys: {sorted(paths)}")
    p = Path(value).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_component(kind: str, name: str, config_dir: Path = CONFIG_DIR) -> dict:
    path = config_dir / kind / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (config_dir / kind).glob("*.yaml"))
        raise FileNotFoundError(f"no {kind} config {name!r}; available: {available}")
    cfg = read_yaml(path)
    cfg.setdefault("name", name)
    return cfg


def load_experiment(path_or_name: str | Path, overrides: dict[str, Any] | None = None,
                    config_dir: Path = CONFIG_DIR) -> dict:
    path = Path(path_or_name)
    if not path.suffix:
        path = config_dir / "experiments" / f"{path_or_name}.yaml"
    exp = read_yaml(path)
    exp.setdefault("name", path.stem)
    exp["dataset"] = deep_merge(load_component("datasets", exp["dataset"], config_dir), exp.get("dataset_overrides", {}))
    exp["model"] = deep_merge(load_component("models", exp["model"], config_dir), exp.get("model_overrides", {}))
    exp["representations"] = {r: load_component("representations", r, config_dir) for r in exp["representations"]}
    exp.pop("dataset_overrides", None)
    exp.pop("model_overrides", None)
    return deep_merge(exp, overrides or {})
