import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nodes.media_nodes import FFmpegBatchConvertNode


class TestFFmpegBatchConvertNode(unittest.TestCase):
    def setUp(self):
        self.node = FFmpegBatchConvertNode()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_file = self.root / "sample.mp4"
        self.input_file.write_bytes(b"fake video content")
        self.output_dir = self.root / "output"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_convert_single_file_success(self):
        output_file = self.output_dir / "sample.mp3"

        def fake_run(command, capture_output, text):
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(b"fake audio content")
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch.object(
            self.node, "_resolve_ffmpeg_executable", return_value="ffmpeg"
        ), patch("nodes.media_nodes.subprocess.run", side_effect=fake_run) as mock_run:
            output_files, success_count, fail_count, report_json = self.node.convert(
                input_mode="single",
                source_path=str(self.input_file),
                file_list="",
                glob_pattern="*.mp4",
                recursive=False,
                output_dir=str(self.output_dir),
                output_format="mp3",
                sample_rate=44100,
                channels=2,
                audio_bitrate="192k",
                overwrite="overwrite",
                continue_on_error=True,
                ffmpeg_path="",
            )

        report = json.loads(report_json)
        self.assertEqual(success_count, 1)
        self.assertEqual(fail_count, 0)
        self.assertEqual(output_files, str(output_file))
        self.assertTrue(output_file.exists())
        self.assertEqual(report["items"][0]["status"], "success")
        self.assertEqual(report["success_count"], 1)
        mock_run.assert_called_once()

    def test_convert_skips_when_output_exists(self):
        existing_output = self.output_dir / "sample.mp3"
        existing_output.parent.mkdir(parents=True, exist_ok=True)
        existing_output.write_bytes(b"existing audio")

        with patch.object(
            self.node, "_resolve_ffmpeg_executable", return_value="ffmpeg"
        ), patch("nodes.media_nodes.subprocess.run") as mock_run:
            output_files, success_count, fail_count, report_json = self.node.convert(
                input_mode="single",
                source_path=str(self.input_file),
                file_list="",
                glob_pattern="*.mp4",
                recursive=False,
                output_dir=str(self.output_dir),
                output_format="mp3",
                sample_rate=44100,
                channels=2,
                audio_bitrate="192k",
                overwrite="skip",
                continue_on_error=True,
                ffmpeg_path="",
            )

        report = json.loads(report_json)
        self.assertEqual(output_files, "")
        self.assertEqual(success_count, 0)
        self.assertEqual(fail_count, 0)
        self.assertEqual(report["items"][0]["status"], "skipped")
        self.assertEqual(report["items"][0]["reason"], "output exists")
        mock_run.assert_not_called()

    def test_convert_missing_input_reports_failure(self):
        missing_file = self.root / "missing.mp4"

        with patch.object(
            self.node, "_resolve_ffmpeg_executable", return_value="ffmpeg"
        ), patch("nodes.media_nodes.subprocess.run") as mock_run:
            output_files, success_count, fail_count, report_json = self.node.convert(
                input_mode="single",
                source_path=str(missing_file),
                file_list="",
                glob_pattern="*.mp4",
                recursive=False,
                output_dir=str(self.output_dir),
                output_format="mp3",
                sample_rate=44100,
                channels=2,
                audio_bitrate="192k",
                overwrite="overwrite",
                continue_on_error=True,
                ffmpeg_path="",
            )

        report = json.loads(report_json)
        self.assertEqual(output_files, "")
        self.assertEqual(success_count, 0)
        self.assertEqual(fail_count, 1)
        self.assertEqual(report["items"][0]["status"], "failed")
        self.assertIn("input file not found", report["items"][0]["error"])
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
