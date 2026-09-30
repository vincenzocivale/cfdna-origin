"""Run provenance: environment, git state, stable hashes, atomic JSON writes."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from cfdna_origin.config import REPO_ROOT


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_default, sort_keys=False))
    tmp.replace(path)


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_default).encode()).hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain")),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def environment() -> dict:
    import numpy
    import pandas
    import sklearn
    import torch

    env = {"python": platform.python_version(), "platform": platform.platform(), "hostname": socket.gethostname(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": numpy.__version__,
           "pandas": pandas.__version__, "sklearn": sklearn.__version__,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "git": git_state(), "time_utc": now()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env
