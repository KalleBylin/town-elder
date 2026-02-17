"""Tests for the indexing benchmark harness script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HEAVY_BENCH_ENV = "TE_RUN_HEAVY_BENCHMARK"
_HEAVY_FILE_COUNT = "60000"
_SMOKE_FILE_COUNT = "120"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(_repo_root() / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing}" if existing else src_path
    )
    return env


def test_benchmark_harness_smoke_generates_results(tmp_path: Path) -> None:
    json_output = tmp_path / "benchmark.json"
    md_output = tmp_path / "benchmark.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_indexing.py",
            "--files",
            _SMOKE_FILE_COUNT,
            "--workers",
            "1",
            "--batch-size",
            "32",
            "--output-json",
            str(json_output),
            "--output-md",
            str(md_output),
        ],
        cwd=_repo_root(),
        env=_benchmark_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json_output.exists()
    assert md_output.exists()

    payload = json.loads(json_output.read_text())
    assert payload["fixture"]["files"] == int(_SMOKE_FILE_COUNT)
    assert "baseline" in payload
    assert "optimized" in payload
    assert payload["optimized"]["scan"]["throughput_files_per_s"] > 0


@pytest.mark.skipif(
    os.environ.get(_HEAVY_BENCH_ENV) != "1",
    reason="Set TE_RUN_HEAVY_BENCHMARK=1 to run the heavyweight benchmark path.",
)
def test_benchmark_harness_60k_opt_in(tmp_path: Path) -> None:
    """Heavy benchmark should only run when explicitly requested."""
    json_output = tmp_path / "benchmark_60k.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_indexing.py",
            "--files",
            _HEAVY_FILE_COUNT,
            "--workers",
            "2",
            "--skip-rerun",
            "--output-json",
            str(json_output),
        ],
        cwd=_repo_root(),
        env=_benchmark_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(json_output.read_text())
    assert payload["fixture"]["files"] == int(_HEAVY_FILE_COUNT)
