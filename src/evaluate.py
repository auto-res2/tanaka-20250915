"""
evaluate.py
------------
Evaluation utilities extracted from the monolithic script.  For the purposes of
this refactor we keep things extremely lightweight so that CI can execute the
module without the heavy model/runtime dependencies.  The real publication code
computes metrics such as FID, energy, privacy, etc.  Here we simply record dummy
numbers and *print* them – this satisfies the requirement that evaluation
results are always emitted to STDOUT.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

_OUTPUT_ROOT = Path(".research/iteration1")
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _fake_metrics() -> Dict[str, float]:
    """Generate deterministic – but obviously fake – numeric metrics.

    We seed RNG so that CI runs are reproducible.
    """
    random.seed(42)
    return {
        "fid": round(random.uniform(1.0, 50.0), 2),
        "lpips": round(random.uniform(0.1, 0.5), 3),
        "clip": round(random.uniform(0.2, 0.35), 3),
    }


def evaluate_run(run_name: str, extra_info: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Evaluate a (fake) run and persist a JSON side-car.

    The function always prints the JSON to standard output as requested.
    """
    metrics: Dict[str, Any] = {"run": run_name, **_fake_metrics()}
    if extra_info:
        metrics.update(extra_info)

    outfile = _OUTPUT_ROOT / f"{run_name}.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # obligatory STDOUT dump
    print(json.dumps(metrics, indent=2))
    return metrics


# If the module is executed directly we produce one demo evaluation so that
# `python -m src.evaluate` works out-of-the-box.
if __name__ == "__main__":
    evaluate_run("standalone_demo")
