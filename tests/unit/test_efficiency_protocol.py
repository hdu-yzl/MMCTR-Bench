import json
import tempfile
import unittest
from pathlib import Path

import torch

from mmctr.analysis import EfficiencyProtocol, load_efficiency_report, save_efficiency_report
from mmctr.core import ContractError


class EfficiencyProtocolTests(unittest.TestCase):
    def test_cpu_measurement_uses_warmup_counts_parameters_and_round_trips(self):
        model = torch.nn.Linear(4, 3)
        model.bias.requires_grad_(False)
        calls = []
        clock_values = iter((10.0, 12.0))
        protocol = EfficiencyProtocol(
            warmup_steps=2,
            measured_steps=3,
            clock=lambda: next(clock_values),
        )
        report = protocol.measure(
            step=lambda: calls.append("step"),
            examples_per_step=8,
            parameter_source=model,
            device="cpu",
            input_fingerprint="data-v1",
        )

        self.assertEqual(5, len(calls))
        self.assertEqual(15, report.total_parameters)
        self.assertEqual(12, report.trainable_parameters)
        self.assertEqual(2.0, report.total_seconds)
        self.assertAlmostEqual(2_000.0 / 3.0, report.latency_ms)
        self.assertEqual(12.0, report.examples_per_second)
        self.assertIsNone(report.peak_memory_bytes)
        self.assertIsNone(report.accelerator_name)
        self.assertEqual(torch.__version__, report.torch_version)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "efficiency.json"
            save_efficiency_report(path, report)
            self.assertEqual(report, load_efficiency_report(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["report"]["measured_steps"] = 9
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "fingerprint"):
                load_efficiency_report(path)

    def test_invalid_protocol_inputs_are_rejected(self):
        with self.assertRaisesRegex(ContractError, "measured_steps"):
            EfficiencyProtocol(warmup_steps=0, measured_steps=0)
        protocol = EfficiencyProtocol(warmup_steps=0, measured_steps=1)
        with self.assertRaisesRegex(ContractError, "examples_per_step"):
            protocol.measure(lambda: None, 0, torch.nn.Linear(1, 1), "cpu", "data")


if __name__ == "__main__":
    unittest.main()
