"""Guard test: the README coverage badge states the floor pyproject.toml enforces.

The badge is the only place outside ``[tool.coverage.report]`` that names the threshold, so raising
``fail_under`` without touching the README would leave a badge claiming a floor the build no longer
holds. Both numbers are read from disk here; neither is written into this file.
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Percent-encoded in the shield URL: %E2%89%A5 is the >= sign, %25 the percent sign.
BADGE_PATTERN = re.compile(r"img\.shields\.io/badge/coverage-%E2%89%A5(\d+)%25")


def _configured_floor() -> int:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return int(config["tool"]["coverage"]["report"]["fail_under"])


def test_coverage_badge_matches_configured_floor() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    matches = BADGE_PATTERN.findall(readme)

    assert matches, "README.md has no coverage badge. Removing it is a deliberate change, so drop this test with it."

    floor = _configured_floor()
    for badged in matches:
        assert int(badged) == floor, (
            f"README coverage badge claims {badged}% but fail_under in pyproject.toml is {floor}%."
        )


def test_readme_publishes_no_measured_coverage_figure() -> None:
    """A percentage from some past run reads as current and nothing re-measures it."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    without_badge = "\n".join(line for line in readme.splitlines() if not BADGE_PATTERN.search(line))

    stray = re.findall(r"[Cc]overage[^.\n]{0,40}?(\d+(?:\.\d+)?)\s?%", without_badge)

    assert not stray, (
        f"README.md states a measured coverage figure: {stray}. The badge reports the enforced floor instead."
    )
