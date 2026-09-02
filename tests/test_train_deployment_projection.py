from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_deployment_projection.py"
SPEC = importlib.util.spec_from_file_location("train_deployment_projection", SCRIPT)
TRAINER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(TRAINER)


class DeploymentProjectionTrainingTest(unittest.TestCase):
    def test_extracts_only_the_supported_bounded_sensor_surface(self) -> None:
        extractor = TRAINER.DOMExtractor()
        extractor.feed(
            "<title> Judul </title><h1> Heading </h1><h4> Tidak didukung </h4>"
            "<a> Tautan </a><a>" + ("x" * 200) + "</a>"
        )
        extractor.close()
        self.assertEqual("Judul Heading Tautan", extractor.text())
        self.assertEqual(
            extractor.text(),
            TRAINER.deployment_text_from_html(
                "<title> Judul </title><h1> Heading </h1><h4> Tidak didukung </h4>"
                "<a> Tautan </a><a>" + ("x" * 200) + "</a>"
            ),
        )

    def test_url_feature_contract_is_complete_and_deterministic(self) -> None:
        values = TRAINER.url_feature_values(
            "https://promo.example.test/slot?ref=7",
            ["slot", "casino"],
        )
        self.assertEqual(TRAINER.URL_FEATURES, list(values))
        self.assertEqual(1.0, values["url_keyword_count"])
        self.assertEqual(1.0, values["url_has_https"])

    def test_policy_rank_prioritizes_recall_without_spending_fpr_buffer(self) -> None:
        buffered = {
            "accuracy": 0.97,
            "precision": 0.96,
            "recall": 0.96,
            "f1_score": 0.96,
            "false_positive_rate": 0.014,
        }
        unbuffered = {**buffered, "recall": 0.99, "false_positive_rate": 0.051}
        self.assertGreater(TRAINER.policy_rank(buffered), TRAINER.policy_rank(unbuffered))

    def test_normalization_recovers_leetspeak_confusables_and_separators(self) -> None:
        self.assertIn("judi", TRAINER.normalize_model_text("judі", ["judi"]))
        self.assertIn("slot", TRAINER.normalize_model_text("sl0t", ["slot"]))
        self.assertIn("slot", TRAINER.normalize_model_text("s-l-o-t", ["slot"]))

    def test_camouflage_augmentation_is_five_times_the_training_frame(self) -> None:
        _, pd, _, _, _, _, _, _, _ = TRAINER.dependencies()
        frame = pd.DataFrame(
            [{
                "deployment_text": "judi taruhan",
                "has_dom_content": True,
                "label": 1,
            }]
        )
        augmented = TRAINER.augment_training_frame(frame, ["judi"], TRAINER.dependencies())
        self.assertEqual(5, len(augmented))
        self.assertEqual({1}, set(augmented["label"]))

    def test_empty_metric_slice_is_pending(self) -> None:
        self.assertEqual("pending", TRAINER.metric_summary([], [])[
            "status"
        ])


if __name__ == "__main__":
    unittest.main()
