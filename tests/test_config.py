import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from cfdna_origin.config import REPO_ROOT, load_component, load_experiment, load_paths, resolve_path
from cfdna_origin.experiments.provenance import stable_hash, write_json


def _paths_dir(tmp_path, base, local=None):
    d = tmp_path / "configs"; d.mkdir()
    (d / "paths.yaml").write_text(yaml.safe_dump(base))
    if local is not None:
        (d / "paths.local.yaml").write_text(yaml.safe_dump(local))
    return d


def test_load_paths_references_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDNA_TEST_ROOT", str(tmp_path / "env_root"))
    monkeypatch.delenv("CFDNA_UNSET_VAR", raising=False)
    d = _paths_dir(tmp_path, {"cache_root": "{data_root}/cache", "data_root": "${CFDNA_TEST_ROOT}/data",
                              "outputs_root": "${CFDNA_UNSET_VAR:-/scratch/out}", "rel": "some/rel"})
    p = load_paths(d)
    assert p["data_root"] == tmp_path / "env_root" / "data"
    assert p["cache_root"] == tmp_path / "env_root" / "data" / "cache"
    assert p["outputs_root"] == Path("/scratch/out")
    assert p["rel"] == REPO_ROOT / "some" / "rel"  # relative paths resolve against the repository root


def test_load_paths_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("CFDNA_UNSET_VAR", raising=False)
    with pytest.raises(ValueError, match="unresolvable"):
        load_paths(_paths_dir(tmp_path, {"a": "{b}/x", "b": "{a}/y"}))
    (tmp_path / "e").mkdir()
    with pytest.raises(KeyError, match="CFDNA_UNSET_VAR"):
        load_paths(_paths_dir(tmp_path / "e", {"a": "${CFDNA_UNSET_VAR}"}))


def test_paths_local_overrides(tmp_path):
    d = _paths_dir(tmp_path, {"data_root": "data", "cache_root": "{data_root}/cache"}, local={"data_root": "/big/disk"})
    p = load_paths(d)
    assert p["data_root"] == Path("/big/disk") and p["cache_root"] == Path("/big/disk/cache")


def test_resolve_path(tmp_path):
    paths = {"data_root": tmp_path}
    assert resolve_path("{data_root}/x/y", paths) == tmp_path / "x" / "y"
    with pytest.raises(KeyError, match="unknown path reference"):
        resolve_path("{nope}/x", paths)


@pytest.mark.parametrize("name", ["gse149438_main", "hra003209_main"])
def test_load_real_experiments(name):
    exp = load_experiment(name)
    assert exp["name"] == name and isinstance(exp["dataset"], dict) and isinstance(exp["model"], dict)
    assert exp["dataset"]["name"] == name.split("_")[0]
    assert exp["model"]["name"] == "hierarchical_mil"
    assert set(exp["representations"]) >= {"functional", "methylation_only", "random", "position_only"}
    assert all(r["name"] == k and "kind" in r for k, r in exp["representations"].items())
    assert exp["dataset"]["label_scheme"] in exp["dataset"]["label_schemes"]
    ov = load_experiment(name, overrides={"training": {"max_epochs": 2}})
    assert ov["training"]["max_epochs"] == 2 and ov["training"]["lr"] == exp["training"]["lr"]


def test_experiment_component_overrides(tmp_path):
    d = tmp_path / "configs"
    for sub in ("datasets", "models", "representations", "experiments"):
        (d / sub).mkdir(parents=True)
    (d / "datasets" / "ds.yaml").write_text(yaml.safe_dump({"split": {"seed": 1, "n_folds": 5}}))
    (d / "models" / "m.yaml").write_text(yaml.safe_dump({"d_model": 64}))
    (d / "representations" / "r.yaml").write_text(yaml.safe_dump({"kind": "random"}))
    (d / "experiments" / "e.yaml").write_text(yaml.safe_dump({
        "dataset": "ds", "model": "m", "representations": ["r"], "dataset_overrides": {"split": {"n_folds": 3}}}))
    exp = load_experiment("e", config_dir=d)
    assert exp["dataset"] == {"name": "ds", "split": {"seed": 1, "n_folds": 3}} and "dataset_overrides" not in exp
    assert exp["representations"]["r"] == {"kind": "random", "name": "r"}


def test_unknown_component_lists_available():
    with pytest.raises(FileNotFoundError) as err:
        load_component("representations", "does_not_exist")
    assert "functional" in str(err.value) and "methylation_only" in str(err.value)


def test_provenance_helpers(tmp_path):
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    write_json(tmp_path / "x.json", {"n": np.int64(3), "arr": np.arange(2), "p": tmp_path})
    assert json.loads((tmp_path / "x.json").read_text()) == {"n": 3, "arr": [0, 1], "p": str(tmp_path)}
    assert not list(tmp_path.glob("*.tmp"))
