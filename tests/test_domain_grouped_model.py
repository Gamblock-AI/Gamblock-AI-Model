from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


ROOT = pathlib.Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_domain_grouped_model.py"
MODULE = load_module("evaluate_domain_grouped_model", SCRIPT)
GROUPED_SPLIT = load_module("grouped_split", ROOT / "scripts" / "grouped_split.py")


class DomainGroupedModelTest(unittest.TestCase):
    def test_connected_grouping_keeps_duplicate_text_and_domain_together(self) -> None:
        rows = [
            {"id": "A", "url": "https://one.example.com/a", "label": "1", "text_clean": "judi"},
            {"id": "B", "url": "https://two.example.net/b", "label": "1", "text_clean": "judi"},
            {"id": "C", "url": "https://other-one.com/c", "label": "1", "text_clean": "different"},
            {"id": "D", "url": "https://other-two.org/d", "label": "0", "text_clean": "different"},
            {"id": "E", "url": "https://safe.example.edu/e", "label": "0", "text_clean": "school"},
            {"id": "F", "url": "https://safe.another.net/f", "label": "0", "text_clean": "school"},
            {"id": "G", "url": "https://third-g.io/g", "label": "1", "text_clean": "casino"},
            {"id": "H", "url": "https://fourth-co.co/h", "label": "1", "text_clean": "betting"},
            {"id": "I", "url": "https://safe-two.io/i", "label": "0", "text_clean": "lesson"},
            {"id": "J", "url": "https://safe-three.co/j", "label": "0", "text_clean": "homework"},
        ]
        group_ids, conflicts = GROUPED_SPLIT.build_group_ids(
            rows,
            {row["id"]: row["text_clean"] for row in rows},
        )
        self.assertEqual(group_ids["A"], group_ids["B"])
        self.assertEqual(group_ids["C"], group_ids["D"])
        self.assertEqual(1, len(conflicts))

        eligible, _ = GROUPED_SPLIT.remove_conflicting_groups(rows, group_ids, conflicts)
        train, test, assignments = GROUPED_SPLIT.stratified_group_split(
            eligible,
            group_ids,
            0.5,
            42,
        )
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertEqual(set(row["id"] for row in eligible), set(assignments))
        self.assertFalse(
            {group_ids[row["id"]] for row in train}
            & {group_ids[row["id"]] for row in test}
        )

    def test_domain_group_normalizes_www_and_subdomains(self) -> None:
        self.assertEqual(
            GROUPED_SPLIT.domain_group("https://www.shop.example.com/path"),
            "domain:example.com",
        )

    def test_group_split_is_deterministic_and_has_no_overlap(self) -> None:
        rows = [
            {"id": "A1", "url": "https://a.example.com", "label": "0"},
            {"id": "A2", "url": "https://sub.a.example.com", "label": "0"},
            {"id": "B1", "url": "https://b.example.org", "label": "0"},
            {"id": "C1", "url": "https://c.example.net", "label": "1"},
            {"id": "D1", "url": "https://d.example.edu", "label": "1"},
            {"id": "E1", "url": "https://e.example.id", "label": "1"},
        ]
        group_ids, conflicts = GROUPED_SPLIT.build_group_ids(rows, {})
        self.assertFalse(conflicts)
        train, test, assignments = GROUPED_SPLIT.stratified_group_split(rows, group_ids, 0.34, 42)
        train_groups = {GROUPED_SPLIT.domain_group(row["url"], row["id"]) for row in train}
        test_groups = {GROUPED_SPLIT.domain_group(row["url"], row["id"]) for row in test}
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(
            assignments,
            GROUPED_SPLIT.stratified_group_split(rows, group_ids, 0.34, 42)[2],
        )

    def test_conflicting_group_is_excluded_from_numeric_evaluation(self) -> None:
        rows = [
            {"id": "A1", "url": "https://same.example.com/a", "label": "0"},
            {"id": "A2", "url": "https://same.example.com/b", "label": "1"},
            {"id": "B1", "url": "https://other.example.net", "label": "0"},
        ]
        group_ids, conflicts = GROUPED_SPLIT.build_group_ids(
            rows,
            {row["id"]: "" for row in rows},
        )
        eligible, conflicts = GROUPED_SPLIT.remove_conflicting_groups(rows, group_ids, conflicts)
        self.assertEqual(["B1"], [row["id"] for row in eligible])
        self.assertEqual(1, len(conflicts))
        self.assertEqual(2, conflicts[0]["rows"])

    def test_empty_slice_is_pending(self) -> None:
        result = MODULE.metric_summary([], [])
        self.assertEqual("pending", result["status"])

    def test_metric_gate_has_required_boundary(self) -> None:
        result = MODULE.metric_summary(
            [1] * 90 + [0] * 10,
            [1] * 90 + [0] * 10,
        )
        self.assertEqual("passed", result["status"])
        self.assertTrue(result["numeric_gate_passed"])
        self.assertFalse(result["gates"]["pkm_progress_v5"]["passed"])

    def test_camouflage_variants_are_deterministic_and_label_independent(self) -> None:
        original = "Judi taruhan online"
        trainer = MODULE.load_trainer()
        for variant in MODULE.CAMOUFLAGE_VARIANTS:
            transformed = MODULE.camouflage_text(original, variant)
            self.assertEqual(transformed, MODULE.camouflage_text(original, variant))
            self.assertEqual(transformed, trainer.camouflage_text(original, variant))
            self.assertNotEqual(original, transformed)

    def test_unicode_confusable_is_normalized_by_the_deployment_contract(self) -> None:
        trainer = MODULE.load_trainer()
        original = trainer.normalize_model_text("Judi taruhan online", ["judi"])
        transformed = trainer.camouflage_text("Judi taruhan online", "unicode_confusable")
        self.assertNotEqual("Judi taruhan online", transformed)
        self.assertEqual(original, trainer.normalize_model_text(transformed, ["judi"]))

    def test_threshold_sensitivity_reports_selected_threshold(self) -> None:
        result = MODULE.threshold_sensitivity(
            [0, 0, 1, 1],
            [0.10, 0.30, 0.70, 0.90],
            [0.10, 0.30, 0.70, 0.90],
            [0.0, 0.0, 1.0, 1.0],
            [True, True, True, True],
            {"ml_weight": 0.5, "rule_weight": 0.5, "threshold": 0.50},
        )
        self.assertEqual(0.50, result["selected_threshold"])
        self.assertEqual(len(MODULE.THRESHOLD_GRID), len(result["results"]))
        self.assertTrue(any(item["selected"] for item in result["results"]))

    def test_calibration_summary_reports_bins_and_scores(self) -> None:
        result = MODULE.calibration_summary([0, 1], [0.10, 0.90], bin_count=2)
        self.assertEqual("reported", result["status"])
        self.assertEqual(2, result["samples"])
        self.assertEqual(2, len(result["bins"]))
        self.assertGreaterEqual(result["brier_score"], 0.0)
        self.assertGreaterEqual(result["expected_calibration_error"], 0.0)

    def test_onnx_export_matches_python_probability_and_decision(self) -> None:
        trainer = MODULE.load_trainer()
        bundle = trainer.dependencies()
        _, pd, _, _, _, _, _, _, _ = bundle
        frame = pd.DataFrame(
            [
                {"deployment_text": "judi taruhan", "has_dom_content": True, "label": 1,
                 **trainer.url_feature_values("https://promo.example.test/judi", ["judi"])},
                {"deployment_text": "materi belajar", "has_dom_content": True, "label": 0,
                 **trainer.url_feature_values("https://school.example.test/course", ["judi"])},
                {"deployment_text": "judi online", "has_dom_content": True, "label": 1,
                 **trainer.url_feature_values("https://play.example.test/online", ["judi"])},
                {"deployment_text": "catatan sekolah", "has_dom_content": True, "label": 0,
                 **trainer.url_feature_values("https://school.example.test/note", ["judi"])},
            ]
        )
        pipeline = trainer.build_pipeline(
            bundle,
            {"max_features": 1000, "min_df": 1, "c": 0.05},
        )
        trainer.fit_pipeline(pipeline, frame)
        with tempfile.TemporaryDirectory() as temporary:
            onnx_path = pathlib.Path(temporary) / "candidate.onnx"
            trainer.export_onnx(pipeline, onnx_path)
            result = MODULE.onnx_parity(pipeline, frame, onnx_path, threshold=0.45)
        self.assertEqual("passed", result["status"], result)
        self.assertTrue(result["prediction_match"])
        self.assertLessEqual(result["max_probability_absolute_error"], result["tolerance"])

    def test_duplicate_audit_detects_cross_split_duplicates(self) -> None:
        rows = [
            {"id": "A", "url_clean": "same", "text_clean": "one", "text_combined": "one"},
            {"id": "B", "url_clean": " SAME  ", "text_clean": "two", "text_combined": "two"},
        ]
        result = MODULE.duplicate_leakage_audit(rows, {"A": "train", "B": "test"})
        self.assertEqual(1, result["normalized_url"]["cross_split_duplicate_groups"])
        self.assertFalse(result["audit_passed"])

    def test_split_integrity_audit_detects_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            clean_path = directory / "clean.csv"
            train_path = directory / "train.csv"
            test_path = directory / "test.csv"
            clean_path.write_text("id,label\nA,0\nB,1\nC,0\n", encoding="utf-8")
            train_path.write_text("id,label\nA,0\nB,1\n", encoding="utf-8")
            test_path.write_text("id,label\nC,0\n", encoding="utf-8")
            manifest = {
                "source": {"dataset_clean_sha256": MODULE.sha256(clean_path)},
                "train": {"sha256": MODULE.sha256(train_path)},
                "test": {"sha256": MODULE.sha256(test_path)},
                "eligible": {"rows": 3},
                "conflicting_groups_excluded": {"rows": 0},
                "assignment_sha256": MODULE.assignment_hash({"A": "train", "B": "train", "C": "test"}),
            }
            test_path.write_text("id,label\nB,1\n", encoding="utf-8")
            result = MODULE.split_integrity_audit(
                clean_path,
                train_path,
                test_path,
                [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                [{"id": "A"}, {"id": "B"}],
                [{"id": "B"}],
                manifest,
            )
        self.assertEqual("failed", result["status"])
        self.assertIn("test_sha256_matches_manifest", result["failed_checks"])
        self.assertIn("train_test_ids_disjoint", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
