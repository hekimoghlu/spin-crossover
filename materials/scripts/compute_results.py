#!/usr/bin/env python3
"""Compute all reimplemented quantities and write results/results.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scmaterials import pipeline  # noqa: E402


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == "__main__":
    R = pipeline.compute_all()
    out = ROOT / "results" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_clean(R), indent=1))
    print("wrote", out)
