"""Run or resume all 27 formal QK-LOFT experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTS = ("principal", "grad_svd", "skew_grad")
SIDES = ("left", "right", "bilateral")
TASKS = ("dtd", "eurosat", "dsprites_orientation")
EPOCHS = 100
SEED = 42


def result_path(support: str, side: str, task: str) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / f"qk_loft_{support}_{side}"
        / "development_800_200"
        / task
        / f"seed_{SEED}.json"
    )


def is_complete(path: Path, support: str, side: str, task: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("epochs") == EPOCHS
        and payload.get("support_type") == support
        and payload.get("side") == side
        and payload.get("task") == task
        and payload.get("seed") == SEED
        and payload.get("test", {}).get("total", 0) > 0
    )


def main() -> None:
    jobs = [
        (support, side, task)
        for support in SUPPORTS
        for side in SIDES
        for task in TASKS
    ]
    logs = PROJECT_ROOT / "results" / "logs" / "qk_loft_factorial"
    logs.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    for index, (support, side, task) in enumerate(jobs, start=1):
        destination = result_path(support, side, task)
        label = f"{support}/{side}/{task}"
        if is_complete(destination, support, side, task):
            print(f"[{index:02d}/{len(jobs)}] SKIP complete {label}", flush=True)
            continue

        log_path = logs / f"{support}_{side}_{task}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.qk_loft.run_qk_loft",
            "--support",
            support,
            "--side",
            side,
            "--task",
            task,
            "--epochs",
            str(EPOCHS),
            "--warmup-epochs",
            "10",
            "--num-workers",
            "4",
            "--seed",
            str(SEED),
        ]
        print(f"[{index:02d}/{len(jobs)}] START {label}", flush=True)
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.perf_counter() - start
        if completed.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print("\n".join(tail), file=sys.stderr, flush=True)
            raise SystemExit(
                f"[{index:02d}/{len(jobs)}] FAILED {label} after {elapsed:.1f}s; "
                f"see {log_path}"
            )
        if not is_complete(destination, support, side, task):
            raise RuntimeError(f"Run finished but result validation failed: {destination}")
        payload = json.loads(destination.read_text(encoding="utf-8"))
        accuracy = 100.0 * float(payload["test"]["accuracy"])
        print(
            f"[{index:02d}/{len(jobs)}] DONE  {label}: "
            f"accuracy={accuracy:.2f}%, elapsed={elapsed:.1f}s",
            flush=True,
        )

    print(
        f"All formal QK-LOFT jobs complete in {time.perf_counter() - total_start:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
