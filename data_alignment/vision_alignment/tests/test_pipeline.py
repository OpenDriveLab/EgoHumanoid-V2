import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from visual_align import pipeline


class PipelineStages(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.video = self.root / "recording.mp4"
        self.motion = self.root / "motion.npz"
        runtime = self.root / "runtime"
        for path in (
            self.video,
            self.motion,
            runtime / "models/sam3.1_multiplex.pt",
            runtime / "models/g1/g1_revo2.xml",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"input")
        self.argv = [
            "eh2-visual-align",
            "--video",
            str(self.video),
            "--motion",
            str(self.motion),
            "--runtime",
            str(runtime),
            "--output",
            str(self.output),
        ]
        self.calls = []

    def command(self, command, **kwargs):
        self.calls.append(command)
        if "visual_align.render" in command:
            (self.output / "aligned.mp4").write_bytes(b"rendered")

    def run_stage(self, stage):
        with (
            patch.object(sys, "argv", [*self.argv, "--stage", stage]),
            patch.object(pipeline.subprocess, "run", side_effect=self.command),
        ):
            pipeline.main()

    def test_full_run_routes_inputs_and_resumes_without_rerunning(self):
        self.run_stage("all")
        self.assertEqual(len(self.calls), 5)
        state = json.loads((self.output / "run.json").read_text())
        self.assertEqual(
            state["completed"], ["segment", "geometry", "inpaint", "render"]
        )
        for command in self.calls:
            if "--video" in command:
                self.assertEqual(command[command.index("--video") + 1], str(self.video))
            if "--joints" in command:
                self.assertEqual(
                    command[command.index("--joints") + 1], str(self.motion)
                )
        self.run_stage("all")
        self.assertEqual(len(self.calls), 5)
        self.video.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Inputs or settings changed"):
            self.run_stage("all")
        self.assertEqual(len(self.calls), 5)

    def test_missing_prerequisites_fail_before_starting_work(self):
        for stage in ("geometry", "inpaint", "render"):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(ValueError, "requires completed stages"):
                    self.run_stage(stage)
                self.assertFalse(self.output.exists())
        self.assertEqual(self.calls, [])

    def test_geometry_and_inpaint_are_independent_after_segmentation(self):
        self.run_stage("segment")
        self.run_stage("inpaint")
        with self.assertRaisesRegex(ValueError, "requires completed stages: geometry"):
            self.run_stage("render")
        self.run_stage("geometry")
        self.run_stage("render")
        self.assertEqual(len(self.calls), 5)
        self.assertTrue((self.output / "aligned.mp4").exists())


if __name__ == "__main__":
    unittest.main()
