import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


class BrandGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.brief = {
            "industry": "친환경 화장품",
            "target": "20-30대 여성",
            "keywords": ["자연", "순수", "건강"],
            "tone": "따뜻하고 신뢰감 있는",
            "competitors": ["경쟁사 A"],
            "notes": "지속 가능한 루틴 강조",
        }

    def test_validate_brief_accepts_required_and_optional_fields(self) -> None:
        main.validate_brief(self.brief)

    def test_validate_brief_rejects_missing_required_field(self) -> None:
        invalid = dict(self.brief)
        del invalid["industry"]
        with self.assertRaisesRegex(ValueError, "industry"):
            main.validate_brief(invalid)

    def test_naming_requires_four_candidates(self) -> None:
        response = {
            "naming_candidates": [
                {"name": f"브랜드 {number}", "meaning": "의미"} for number in range(1, 5)
            ]
        }
        with patch("main.call_json_chat", return_value=response):
            candidates = main.generate_naming(object(), self.brief)
        self.assertEqual(len(candidates), 4)

    def test_story_is_limited_to_300_characters(self) -> None:
        with patch("main.call_json_chat", return_value={"story": "가" * 301}):
            story = main.generate_story(object(), self.brief, "브랜드", "의미", "슬로건")
        self.assertEqual(len(story), 300)

    def test_logo_failure_does_not_stop_next_concept(self) -> None:
        png = base64.b64encode(b"test-png").decode("ascii")

        class FakeImages:
            calls = 0

            def generate(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary image error")
                return SimpleNamespace(data=[SimpleNamespace(b64_json=png)])

        client = SimpleNamespace(images=FakeImages())
        colors = {
            "main_color": {"name": "Green", "hex": "#2E7D32", "reason": "main"},
            "sub_colors": [
                {"name": "Light", "hex": "#81C784", "reason": "sub"},
                {"name": "Pale", "hex": "#E8F5E9", "reason": "sub"},
                {"name": "Cream", "hex": "#FFF8E1", "reason": "sub"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            files, errors = main.generate_logos(
                client, self.brief, "브랜드", "슬로건", colors, Path(directory)
            )
            self.assertEqual(files, ["logo_02.png"])
            self.assertEqual(len(errors), 1)
            self.assertTrue((Path(directory) / "logo_02.png").exists())


if __name__ == "__main__":
    unittest.main()
