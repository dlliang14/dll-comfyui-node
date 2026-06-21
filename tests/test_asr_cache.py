import json
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nodes.asr_nodes import ParaformerBatchASRNode


class ParaformerBatchASRCacheTests(unittest.TestCase):
    def test_presigned_query_does_not_change_cache_key(self):
        cache_dir = Path("cache")
        first = ParaformerBatchASRNode._cache_path(
            cache_dir,
            "https://example.com/audio/a.m4a?signature=one",
            "paraformer-v2",
            ["zh"],
        )
        second = ParaformerBatchASRNode._cache_path(
            cache_dir,
            "https://EXAMPLE.com/audio/a.m4a?signature=two",
            "paraformer-v2",
            ["zh"],
        )
        self.assertEqual(first, second)

    def test_cache_only_returns_text_without_api_key(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            url = "https://example.com/audio/a.m4a?signature=temporary"
            cache_path = ParaformerBatchASRNode._cache_path(
                cache_dir, url, "paraformer-v2", ["zh"]
            )
            ParaformerBatchASRNode._write_cache(
                cache_path,
                {
                    "version": 1,
                    "created_at": "2026-06-21T00:00:00+00:00",
                    "source_url": "https://example.com/audio/a.m4a",
                    "model": "paraformer-v2",
                    "language_hints": ["zh"],
                    "text": "缓存的录音文字",
                    "transcription_url": "https://example.com/result.json",
                    "task_id": "cached-task",
                },
            )

            texts, result_urls, report_raw = ParaformerBatchASRNode().transcribe(
                audio_urls=url,
                language_hints="zh",
                model="paraformer-v2",
                continue_on_error=False,
                poll_interval_sec=1,
                timeout_sec=30,
                api_key="",
                cache_dir=str(cache_dir),
                cache_mode="cache_only",
            )

            report = json.loads(report_raw)
            self.assertEqual(texts, "缓存的录音文字")
            self.assertEqual(result_urls, "https://example.com/result.json")
            self.assertEqual(report["cache_hit_count"], 1)
            self.assertEqual(report["asr_request_count"], 0)
            self.assertEqual(report["items"][0]["status"], "cached")

    def test_cache_only_reports_miss_without_calling_asr(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            texts, result_urls, report_raw = ParaformerBatchASRNode().transcribe(
                audio_urls="https://example.com/audio/missing.m4a",
                language_hints="zh",
                model="paraformer-v2",
                continue_on_error=True,
                poll_interval_sec=1,
                timeout_sec=30,
                api_key="",
                cache_dir=temporary_dir,
                cache_mode="cache_only",
            )

            report = json.loads(report_raw)
            self.assertEqual(texts, "")
            self.assertEqual(result_urls, "")
            self.assertEqual(report["cache_hit_count"], 0)
            self.assertEqual(report["asr_request_count"], 0)
            self.assertEqual(report["fail_count"], 1)
            self.assertIn("ASR cache miss", report["items"][0]["error"])

    def test_successful_asr_result_is_written_and_reused(self):
        class FakeTranscription:
            call_count = 0

            @classmethod
            def async_call(cls, **_kwargs):
                cls.call_count += 1
                return {"output": {"task_id": "new-task"}}

        with tempfile.TemporaryDirectory() as temporary_dir:
            node = ParaformerBatchASRNode()
            url = "https://example.com/audio/new.m4a?signature=first"
            result = SimpleNamespace(status_code=HTTPStatus.OK)

            with patch.object(
                node, "_load_transcription_api", return_value=FakeTranscription
            ), patch.object(
                node, "_wait_result", return_value=result
            ), patch.object(
                node,
                "_extract_text_and_url",
                return_value=("新的转写文本", "https://example.com/result.json"),
            ):
                texts, _, report_raw = node.transcribe(
                    audio_urls=url,
                    language_hints="zh",
                    model="paraformer-v2",
                    continue_on_error=False,
                    poll_interval_sec=1,
                    timeout_sec=30,
                    api_key="key",
                    cache_dir=temporary_dir,
                    cache_mode="use_cache",
                )

            first_report = json.loads(report_raw)
            cache_path = Path(first_report["items"][0]["cache_path"])
            self.assertEqual(texts, "新的转写文本")
            self.assertTrue(cache_path.is_file())
            self.assertEqual(FakeTranscription.call_count, 1)

            cached_texts, _, cached_report_raw = node.transcribe(
                audio_urls="https://example.com/audio/new.m4a?signature=renewed",
                language_hints="zh",
                model="paraformer-v2",
                continue_on_error=False,
                poll_interval_sec=1,
                timeout_sec=30,
                api_key="",
                cache_dir=temporary_dir,
                cache_mode="cache_only",
            )

            cached_report = json.loads(cached_report_raw)
            self.assertEqual(cached_texts, "新的转写文本")
            self.assertEqual(cached_report["cache_hit_count"], 1)
            self.assertEqual(FakeTranscription.call_count, 1)


if __name__ == "__main__":
    unittest.main()
