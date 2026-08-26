from __future__ import annotations

import traceback
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_audit.common import (
    AuditSection,
    configure_utf8_stdout,
    load_json,
    make_report_dir,
    resolve_project_path,
    write_json,
)
from pipeline_audit.audits import (
    batching_audit,
    bin_audit,
    checkpoint_audit,
    model_audit,
)


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def run_stage(name, function, config, report_dir) -> AuditSection:
    print(f"[{name}] running")
    try:
        section = function(config, report_dir)
    except Exception:
        section = AuditSection(name)
        section.add(
            "uncaught_exception",
            "fail",
            traceback.format_exc(),
        )
    counts = {status: 0 for status in ("pass", "warn", "fail", "skip")}
    for check in section.checks:
        counts[check["status"]] += 1
    print(
        f"[{name}] "
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )
    return section


def main() -> None:
    configure_utf8_stdout()
    config = load_json(CONFIG_PATH)
    report_root = resolve_project_path(config["paths"]["report_root"])
    report_dir = make_report_dir(report_root)

    stages = [
        ("batching", batching_audit.run),
        ("model", model_audit.run),
        ("bin", bin_audit.run),
        ("checkpoint", checkpoint_audit.run),
    ]
    sections = [
        run_stage(name, function, config, report_dir)
        for name, function in stages
    ]

    totals = {status: 0 for status in ("pass", "warn", "fail", "skip")}
    for section in sections:
        for check in section.checks:
            totals[check["status"]] += 1

    if totals["fail"]:
        result = "FAIL"
    elif totals["skip"]:
        result = "INCOMPLETE"
    else:
        result = "PASS"

    summary = {
        "project_root": str(PROJECT_ROOT),
        "config": str(CONFIG_PATH),
        "report_dir": str(report_dir),
        "result": result,
        "totals": totals,
        "sections": [section.to_dict() for section in sections],
    }
    write_json(report_dir / "summary.json", summary)

    print(f"report: {report_dir}")
    print(
        f"result: {result} (warn={totals['warn']}, skip={totals['skip']})"
    )
    if totals["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
