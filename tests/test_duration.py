"""Duration arithmetic checks; run without torch or ComfyUI."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("refpack_duration", next(p for p in (Path(__file__).resolve().parents[2] / "ComfyUI-MiniMax-H3-RefPacks" / "duration.py", Path(__file__).resolve().parents[2] / "BExplains_H3_RefPacks" / "duration.py") if p.is_file()))
duration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(duration)


class DurationTests(unittest.TestCase):
    def test_requested_durations_round_up(self):
        for seconds, expected in [(0.01, 5), (1, 39), (3, 73), (5, 124), (10, 243), (170, 4085)]:
            with self.subTest(seconds=seconds):
                self.assertEqual(duration.h3_duration_frames(seconds), expected)

    def test_valid_boundaries_and_neighbours(self):
        for frames in (5, 22, 39, 73, 124, 243, 4085):
            seconds = frames / 24
            self.assertEqual(duration.h3_duration_frames(seconds), frames)
            self.assertEqual(duration.h3_duration_frames(seconds - 0.000001), frames)
            self.assertEqual(duration.h3_duration_frames(seconds + 0.000001), frames + 17)

    def test_invalid_duration(self):
        for seconds in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                duration.h3_duration_frames(seconds)


if __name__ == "__main__":
    unittest.main()
