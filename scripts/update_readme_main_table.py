"""Render the complete five-seed reference tables into the README body."""

import argparse
import json
from pathlib import Path
import statistics

START = "<!-- MAIN_TABLE_SNAPSHOT_START -->"
END = "<!-- MAIN_TABLE_SNAPSHOT_END -->"
ANCHOR = "## Runs and results\n"
DATASET_LABELS = {
    "antm2c": "AntM2C",
    "microlens": "MicroLens",
    "tiktok": "TikTok",
    "ebnerd": "EB-NeRD",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/main_table_ebnerd/main_table.json")
    )
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    seeds = [2025, 2026, 2027, 2028, 2029]
    if report["plan"]["seeds"] != seeds:
        raise ValueError("snapshot requires all five declared seeds 2025–2029")
    datasets, models, jobs = report["plan"]["datasets"], report["plan"]["models"], report["jobs"]
    if set(datasets) != {"antm2c", "microlens", "tiktok", "ebnerd"} or len(datasets) != 4:
        raise ValueError("this reference snapshot requires the four declared datasets")
    if len(models) != 21 or len(set(models)) != 21:
        raise ValueError("this reference snapshot requires 21 unique models")
    expected = {(d, m, s) for d in datasets for m in models for s in seeds}
    identities = [(j["dataset"], j["model"], j["seed"]) for j in jobs]
    if set(identities) != expected or len(identities) != len(expected):
        raise ValueError("report jobs do not exactly cover the matrix")
    if any(job["status"] != "completed" for job in jobs):
        raise ValueError("snapshot requires every job to be completed")
    lines = [
        START,
        "## Reference results",
        "",
        "The following tables provide **reference results after dataset restructuring**. "
        "They cover four datasets, 21 models and five seeds (2025–2029), totaling 420 runs. "
        "Values are arithmetic mean ± sample standard deviation (`ddof=1`), "
        "recomputed from all per-seed job results. AUC ↑; LogLoss ↓.",
        "",
        "AntM2C / MicroLens / TikTok use canonical-v1; EB-NeRD uses canonical-v4 "
        "(history length 5, ID width 16, 10% of local-test impressions). "
        "See the [benchmark protocol](docs/benchmark-protocol.md) for dataset and evaluation details.",
        "",
    ]
    for dataset in datasets:
        lines.extend(
            [
                "### " + DATASET_LABELS[dataset],
                "",
                "| Model | AUC ↑ (mean ± std) | LogLoss ↓ (mean ± std) |",
                "|---|---:|---:|",
            ]
        )
        for model in models:
            selected = [j for j in jobs if j["dataset"] == dataset and j["model"] == model]
            cells = []
            row = next(
                r for r in report["table"] if r["dataset"] == dataset and r["model"] == model
            )
            if row["n_success"] != 5 or row["std_ddof"] != 1:
                raise ValueError("incomplete aggregate")
            for metric in ("test_auc", "test_loss"):
                values = [float(j["metrics"][metric]) for j in selected]
                mean, std = statistics.mean(values), statistics.stdev(values)
                if (
                    abs(mean - row[metric + "_mean"]) > 1e-12
                    or abs(std - row[metric + "_std"]) > 1e-12
                ):
                    raise ValueError("aggregate disagrees with raw seeds")
                cells.append("{:.6f} ± {:.6f}".format(mean, std))
            lines.append("| {} | {} | {} |".format(model, *cells))
        lines.append("")
    lines.extend(
        [
            "Source: `outputs/main_table_ebnerd/main_table.json`. Regenerate the tables with "
            '`"$MMCTR_PYTHON" scripts/update_readme_main_table.py`.',
            END,
        ]
    )
    text = args.readme.read_text(encoding="utf-8")
    snapshot = "\n".join(lines)
    if text.count(START) != text.count(END) or text.count(START) > 1:
        raise ValueError("README must contain at most one complete table marker pair")
    if START in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + snapshot + after
    else:
        if text.count(ANCHOR) != 1:
            raise ValueError("README must contain exactly one Runs and results section")
        text = text.replace(ANCHOR, snapshot + "\n\n" + ANCHOR, 1)
    args.readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
