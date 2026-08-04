import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mmctr.analysis import (
    load_standard_results,
    render_metric_figure,
    save_figure_provenance,
)
from mmctr.core import ContractError


def _result(path, run_id, auc, status="completed", model="dnn"):
    payload = {
        "schema_version": 1,
        "task_id": run_id,
        "task_key": "key-" + run_id,
        "run_id": run_id,
        "status": status,
        "dataset": "antm2c",
        "model": model,
        "seed": 7,
        "device": "cpu",
        "data_fingerprint": "data-v1",
        "metrics": {"val_auc": auc},
        "artifact_dir": None,
        "error": None,
        "metadata": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class PlottingProtocolTests(unittest.TestCase):
    def test_metric_figure_is_rendered_from_standard_results_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            _result(first, "run-1", 0.7, model="dnn")
            _result(second, "run-2", 0.8, model="dcn")
            figure_path = root / "metric.png"

            provenance = render_metric_figure(
                (first, second),
                output_path=figure_path,
                metric="val_auc",
                kind="bar",
                group_by="model",
                title="Validation AUC",
            )

            self.assertTrue(figure_path.is_file())
            self.assertGreater(figure_path.stat().st_size, 0)
            provenance_path = root / "metric.png.provenance.json"
            self.assertTrue(provenance_path.is_file())
            self.assertEqual("val_auc", provenance["figure_config"]["metric"])
            self.assertEqual(["dcn", "dnn"], provenance["figure_config"]["groups"])

    def test_legacy_hard_coded_plot_scripts_are_removed(self):
        legacy_plot = Path(__file__).resolve().parents[2] / "src/analysis/plot"
        self.assertEqual([], sorted(legacy_plot.glob("*.py")))

    def test_standard_results_and_figure_provenance_are_source_backed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            _result(first, "run-1", 0.7)
            _result(second, "run-2", 0.8)

            rows = load_standard_results((first, second), required_metrics=("val_auc",))
            self.assertEqual(("run-1", "run-2"), tuple(row.run_id for row in rows))
            provenance_path = root / "figure.provenance.json"
            payload = save_figure_provenance(
                provenance_path,
                (first, second),
                {"metric": "val_auc", "kind": "bar"},
                script_version="plot-v1",
            )
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).hexdigest(),
                payload["inputs"][0]["sha256"],
            )
            self.assertEqual(64, len(payload["fingerprint"]))

            _result(second, "run-2", 0.8, status="failed")
            with self.assertRaisesRegex(ContractError, "completed"):
                load_standard_results((second,), required_metrics=("val_auc",))


if __name__ == "__main__":
    unittest.main()
