import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from visual_align import segment


class SegmentationSessions(unittest.TestCase):
    def test_windows_cover_short_and_long_recordings(self):
        for n, fps in [(1, 50), (4, 10), (604, 50), (986, 50), (497, 50), (10001, 29.97)]:
            windows = segment.propagation_windows(n, fps)
            covered = set()
            for start, end in windows:
                self.assertTrue(0 <= start <= end < n)
                self.assertLessEqual(end - start, round(5 * fps))
                covered.update(range(start, end + 1))
            self.assertEqual(covered, set(range(n)))

    def test_each_direction_has_a_fresh_prompted_session(self):
        sessions = {}
        closed = []

        class Predictor:
            def handle_request(self, request):
                if request["type"] == "start_session":
                    sid = len(sessions)
                    sessions[sid] = {"resource": request["resource_path"]}
                    return {"session_id": sid}
                state = sessions[request["session_id"]]
                if request["type"] == "add_prompt":
                    state.update(request)
                elif request["type"] == "close_session":
                    closed.append(request["session_id"])

            def handle_stream_request(self, request):
                state = sessions[request["session_id"]]
                assert "used" not in state
                state["used"] = True
                start = request["start_frame_index"]
                assert start == state["frame_index"]
                assert start == 0
                assert request["propagation_direction"] == "forward"
                frames = range(len(list(Path(state["resource"]).glob("*.png"))))
                for i in frames:
                    masks = np.zeros((1, 16, 16), bool)
                    original = int(
                        cv2.imread(str(Path(state["resource"]) / f"{i:06d}.png"))[0, 0, 0]
                    )
                    masks[:, original % 16 : original % 16 + 1, :] = True
                    if state["text"] == "watch":
                        masks[:, (original + 1) % 16, :] = True
                    yield {"frame_index": i, "outputs": {"out_binary_masks": masks}}

        builder = types.ModuleType("sam3.model_builder")
        builder.build_sam3_multiplex_video_predictor = lambda **kwargs: Predictor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "rgb.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10, (16, 16))
            for i in range(85):
                writer.write(np.full((16, 16, 3), i, np.uint8))
            writer.release()
            checkpoint = root / "weights.pt"
            checkpoint.write_bytes(b"test")
            output = root / "out"
            args = [
                "segment",
                "--video",
                str(video),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", args),
                patch.dict(sys.modules, {"sam3.model_builder": builder}),
            ):
                segment.main()
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(len(sessions), 28)
            self.assertEqual(closed, list(range(28)))
            self.assertEqual(report["prompts"]["tracker"]["prompt_frames"], [22, 59, 79, 42])
            self.assertEqual(report["prompts"]["tracker"]["text_prompts"], ["wristband", "watch"])
            for name in ("arm", "glove", "tracker"):
                self.assertEqual(len(list((output / name).glob("*.png"))), 85)
                self.assertEqual(report["prompts"][name]["empty_frames"], 0)
                for i in range(85):
                    actual = cv2.imread(str(output / name / f"{i:06d}.png"), 0)
                    expected = np.zeros((16, 16), np.uint8)
                    expected[i % 16] = 255
                    if name == "tracker":
                        expected[(i + 1) % 16] = 255
                    np.testing.assert_array_equal(actual, expected)
            self.assertFalse(list(output.glob("frames-*")))


if __name__ == "__main__":
    unittest.main()
