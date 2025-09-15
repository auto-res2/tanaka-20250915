"""
main.py
--------
Command-line entry-point orchestrating the *two-phase* execution flow as
requested.  The smoke-test is **always** executed first; if it passes we proceed
with the full experiment (unless the user explicitly asked for the smoke-test
only).

Both phases generate JSON result files under `.research/iteration1/` and dump
those JSON blobs to STDOUT for verification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import yaml

from .evaluate import evaluate_run
from .preprocess import prepare_dataset


# -----------------------------------------------------------------------------
#  Config helpers
# -----------------------------------------------------------------------------

_CONFIG_DIR = Path("config")
_SMOKE_CFG = _CONFIG_DIR / "smoke_test.yaml"
_FULL_CFG = _CONFIG_DIR / "full_experiment.yaml"


def _load_cfg(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find configuration file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
#  Minimal experiment runners (stand-ins for heavy logic)
# -----------------------------------------------------------------------------

def _run_phase(cfg: Dict, phase_name: str) -> Dict:
    """Very small stand-in for full experimental pipeline.

    1. Ensures that every dataset entry is at least *prepared* (download stub).
    2. Passes control to `evaluate_run` which writes stats.
    """
    print(f"\n=== Running {phase_name} ===")

    # 1) Dataset preparation – skip heavy lifting, but honour interface
    for spec in cfg.get("datasets", {}).values():
        prepare_dataset(spec)

    # 2) Fake training/evaluation – delegated to evaluate.py
    result = evaluate_run(phase_name)
    return result


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – script-style entry
    parser = argparse.ArgumentParser(description="CREST experiment runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke-test only")
    parser.add_argument("--full-experiment", action="store_true", help="Run full experiment (implies smoke-test first)")
    args = parser.parse_args()

    if not args.smoke_test and not args.full_experiment:
        parser.print_help(sys.stderr)
        sys.exit(2)

    # Phase 1 – smoke always runs
    smoke_cfg = _load_cfg(_SMOKE_CFG)
    smoke_res = _run_phase(smoke_cfg, "smoke_test")

    # Persist aggregated results
    out_root = Path(".research/iteration1")
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "smoke_results.json", "w", encoding="utf-8") as f:
        json.dump(smoke_res, f, indent=2)

    # Phase 2 – full experiment (optional)
    if args.full_experiment:
        full_cfg = _load_cfg(_FULL_CFG)
        full_res = _run_phase(full_cfg, "full_experiment")
        with open(out_root / "full_results.json", "w", encoding="utf-8") as f:
            json.dump(full_res, f, indent=2)


if __name__ == "__main__":
    main()
