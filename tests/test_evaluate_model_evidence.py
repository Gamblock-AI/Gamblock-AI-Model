import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_model_evidence.py"
SPEC = importlib.util.spec_from_file_location("evaluate_model_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class EvaluateModelEvidenceTest(unittest.TestCase):
    def test_metric_formula_and_targets(self):
        rows = [
            {"label": "1", "prediction": "1"},
            {"label": "1", "prediction": "0"},
            {"label": "0", "prediction": "1"},
            {"label": "0", "prediction": "0"},
        ]
        result = MODULE.metrics(rows)
        self.assertEqual(result["confusion_matrix"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertEqual(result["accuracy"], 0.5)
        self.assertFalse(result["numeric_gate_passed"])

    def test_checked_in_snapshot_is_reproducible_and_provisional(self):
        prediction_path = ROOT.parent / "gamblock-ai-testing/model/private/replay_input/predictions.csv"
        if not prediction_path.exists():
            self.skipTest("private frozen prediction snapshot is not available in this workspace")
        report = MODULE.build_report(ROOT, prediction_path)
        self.assertEqual(report["dataset"]["raw"]["total_rows"], 12964)
        self.assertEqual(report["dataset"]["clean"]["rows"], 12960)
        self.assertEqual(report["dataset"]["train"]["rows"], 10368)
        self.assertEqual(report["dataset"]["test"]["rows"], 2592)
        self.assertEqual(
            report["evaluation"]["all_test_rows"]["confusion_matrix"],
            {"tp": 799, "tn": 1725, "fp": 30, "fn": 38},
        )
        self.assertTrue(report["evaluation"]["all_test_rows"]["numeric_gate_passed"])
        self.assertEqual(report["evidence_maturity"], "provisional")
        self.assertFalse(report["audit"]["passed"])
        self.assertFalse(report["privacy"]["raw_url_or_dom_emitted"])


if __name__ == "__main__":
    unittest.main()
