#!/usr/bin/env python3
"""Merge the SARIF files each analyser writes into one review input.

Every tool that can describe a finding as a location plus a message writes SARIF into `.review/`,
and this collapses them into `.review/merged.sarif`. The point is the review step: reading one
file of located findings alongside a change's spec delta is a different job from reading a raw
diff, because the findings already say which line and which rule, and the delta already says what
was meant to change. Nothing here interprets a finding; it only puts them in one place.

SARIF's own model does the merging: a report is a list of `runs`, each carrying its own tool
metadata, so concatenating the run lists is both legal and lossless. Merging results across runs
instead would strip the rule definitions that give a finding its severity and description.

Usage:
    python scripts/merge_sarif.py [--review-dir .review] [--output .review/merged.sarif]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"


def load_runs(path: Path) -> list[dict[str, Any]]:
    """Read one SARIF file and return its runs.

    A tool that produced no findings still writes a valid report with an empty results list, which
    is worth keeping: an empty run is evidence the tool ran, and dropping it makes "clean" and
    "never executed" look the same to whoever reads the merged file.

    Args:
        path: SARIF file to read.

    Returns:
        The file's runs, or an empty list if it is unreadable or not SARIF-shaped.
    """
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: skipping {path}: {e}", file=sys.stderr)
        return []

    if not isinstance(report, dict):
        print(f"warning: skipping {path}: top level is not a SARIF object", file=sys.stderr)
        return []

    runs = report.get("runs", [])
    if not isinstance(runs, list):
        print(f"warning: skipping {path}: 'runs' is not a list", file=sys.stderr)
        return []
    return runs


def count_results(runs: list[dict[str, Any]]) -> int:
    """Total findings across a list of runs."""
    return sum(len(run.get("results", []) or []) for run in runs)


def tool_name(run: dict[str, Any]) -> str:
    """Best-effort name of the tool that produced a run."""
    driver = run.get("tool", {}).get("driver", {})
    name = driver.get("name")
    return str(name) if name else "unknown"


def merge(review_dir: Path, output: Path) -> int:
    """Merge every SARIF file in `review_dir` into `output`.

    Args:
        review_dir: Directory holding the per-tool SARIF files.
        output: File to write the merged report to.

    Returns:
        Total number of findings in the merged report.
    """
    # The output is excluded by name rather than by writing somewhere else, so that a re-run does
    # not fold the previous merge back into itself and double every finding.
    inputs = sorted(p for p in review_dir.glob("*.sarif") if p.resolve() != output.resolve())

    merged_runs: list[dict[str, Any]] = []
    for path in inputs:
        runs = load_runs(path)
        merged_runs.extend(runs)
        print(f"{path.name}: {len(runs)} run(s), {count_results(runs)} finding(s)")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"$schema": SARIF_SCHEMA, "version": SARIF_VERSION, "runs": merged_runs}, indent=2) + "\n"
    )

    total = count_results(merged_runs)
    tools = ", ".join(sorted({tool_name(run) for run in merged_runs})) or "none"
    print(f"\nwrote {output}: {total} finding(s) from {len(merged_runs)} run(s) [{tools}]")
    return total


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review-dir", type=Path, default=Path(".review"), help="directory holding *.sarif")
    parser.add_argument("--output", type=Path, default=None, help="merged output path")
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit non-zero when the merged report contains any finding",
    )
    args = parser.parse_args()

    review_dir: Path = args.review_dir
    output: Path = args.output or review_dir / "merged.sarif"

    if not review_dir.is_dir():
        print(f"no such directory: {review_dir} (run `make scan` first)", file=sys.stderr)
        return 1

    total = merge(review_dir, output)
    return 1 if (args.fail_on_findings and total) else 0


if __name__ == "__main__":
    sys.exit(main())
