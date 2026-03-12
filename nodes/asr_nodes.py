from __future__ import annotations

import json
import os
import time
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen


class ParaformerBatchASRNode:
    """Batch ASR node powered by DashScope Paraformer v2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "one HTTP/HTTPS URL per line (public or presigned)",
                    },
                ),
                "language_hints": (
                    "STRING",
                    {"default": "zh", "multiline": False},
                ),
                "model": (
                    "STRING",
                    {"default": "paraformer-v2", "multiline": False},
                ),
                "continue_on_error": ("BOOLEAN", {"default": True}),
                "poll_interval_sec": ("INT", {"default": 2, "min": 1, "max": 30}),
                "timeout_sec": ("INT", {"default": 600, "min": 30, "max": 7200}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("texts", "transcription_urls", "report_json")
    FUNCTION = "transcribe"
    CATEGORY = "dlliang14/asr"

    def transcribe(
        self,
        audio_urls: str,
        language_hints: str,
        model: str,
        continue_on_error: bool,
        poll_interval_sec: int,
        timeout_sec: int,
        api_key: str = "",
        oss_config_json: str = "",
        public_base_url: str = "",
        **kwargs: Any,
    ):
        # backward compatibility for old workflows
        _ = (oss_config_json, public_base_url, kwargs)

        try:
            import dashscope
            from dashscope.audio.asr import Transcription
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "dashscope is required. Please install dependencies from requirements.txt"
            ) from exc

        if api_key.strip():
            dashscope.api_key = api_key.strip()
        elif not os.getenv("DASHSCOPE_API_KEY"):
            raise RuntimeError(
                "Missing DashScope API key. Set api_key input or DASHSCOPE_API_KEY."
            )

        resolved_urls = self._parse_audio_urls(audio_urls)

        hints = [
            item.strip()
            for item in language_hints.replace(";", ",").split(",")
            if item.strip()
        ]
        if not hints:
            hints = ["zh"]

        report: Dict[str, Any] = {
            "model": model,
            "language_hints": hints,
            "total_inputs": len(resolved_urls),
            "items": [],
        }
        texts: List[str] = []
        transcription_urls: List[str] = []

        for url in resolved_urls:
            try:
                task_resp = Transcription.async_call(
                    model=model,
                    file_urls=[url],
                    language_hints=hints,
                )

                task_id = self._get_nested(task_resp, "output", "task_id")
                if not task_id:
                    raise RuntimeError(f"Task submission failed, response: {task_resp}")

                result = self._wait_result(
                    Transcription=Transcription,
                    task_id=str(task_id),
                    poll_interval_sec=poll_interval_sec,
                    timeout_sec=timeout_sec,
                )

                if getattr(result, "status_code", None) != HTTPStatus.OK:
                    raise RuntimeError(
                        str(getattr(result, "message", "transcription failed"))
                    )

                text, result_url = self._extract_text_and_url(result)
                if not result_url:
                    raise RuntimeError("Missing transcription_url in ASR response")
                texts.append(text)
                transcription_urls.append(result_url)
                report["items"].append(
                    {
                        "input": url,
                        "status": "success",
                        "text": text,
                        "transcription_url": result_url,
                        "task_id": str(task_id),
                    }
                )
            except Exception as exc:
                report["items"].append(
                    {
                        "input": url,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                if not continue_on_error:
                    raise

        report["success_count"] = len(
            [i for i in report["items"] if i["status"] == "success"]
        )
        report["fail_count"] = len(
            [i for i in report["items"] if i["status"] == "failed"]
        )

        return (
            "\n".join(texts),
            "\n".join(transcription_urls),
            json.dumps(report, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _parse_audio_urls(audio_urls: str) -> List[str]:
        resolved = [line.strip() for line in audio_urls.splitlines() if line.strip()]
        if not resolved:
            return []

        invalid = [
            u
            for u in resolved
            if not (u.startswith("http://") or u.startswith("https://"))
        ]
        if invalid:
            raise ValueError(
                "audio_urls must be public HTTP/HTTPS URLs, one per line. "
                f"Invalid entries: {invalid[:3]}. "
                "If input comes from media node, set oss_output_mode=public_url."
            )
        return resolved

    @staticmethod
    def _wait_result(
        Transcription: Any,
        task_id: str,
        poll_interval_sec: int,
        timeout_sec: int,
    ) -> Any:
        start = time.time()
        while True:
            response = Transcription.fetch(task=task_id)
            status = ParaformerBatchASRNode._get_nested(
                response, "output", "task_status"
            )
            if status in {"SUCCEEDED", "FAILED"}:
                return response

            if (time.time() - start) > timeout_sec:
                raise TimeoutError(f"ASR task timeout: {task_id}")
            time.sleep(poll_interval_sec)

    @staticmethod
    def _extract_text_and_url(response: Any) -> Tuple[str, str]:
        output = ParaformerBatchASRNode._safe_get_attr(response, "output")
        if output is None:
            return "", ""

        payload = ParaformerBatchASRNode._to_mapping(output)

        text = str(payload.get("text", "") or "")
        url = str(
            payload.get("transcription_url")
            or payload.get("url")
            or payload.get("result_url")
            or ""
        )

        # Task-level response schema (contains results list with per-file status/url).
        results = payload.get("results") or payload.get("result")
        if isinstance(results, list) and results:
            first = results[0] if isinstance(results[0], dict) else {}
            subtask_status = str(first.get("subtask_status", "") or "")
            if subtask_status == "FAILED":
                code = str(first.get("code", "") or "")
                message = str(first.get("message", "") or "")
                raise RuntimeError(f"Subtask failed: {code} {message}".strip())

            if not url:
                url = str(
                    first.get("transcription_url")
                    or first.get("url")
                    or first.get("result_url")
                    or ""
                )

        # Detail-response schema might already contain transcripts.
        if not text:
            text = ParaformerBatchASRNode._extract_text_from_transcript_payload(payload)

        if not text:
            sentence_list = payload.get("sentence_list") or payload.get("sentences")
            if isinstance(sentence_list, list):
                parts = []
                for sentence in sentence_list:
                    if isinstance(sentence, dict) and sentence.get("text"):
                        parts.append(str(sentence["text"]))
                text = "".join(parts)

        # If only got transcription_url, fetch detail JSON to extract final text.
        if not text and url:
            detail_payload = ParaformerBatchASRNode._download_transcription_payload(url)
            if detail_payload:
                text = ParaformerBatchASRNode._extract_text_from_transcript_payload(
                    detail_payload
                )

        return text, url

    @staticmethod
    def _get_nested(obj: Any, *keys: str) -> Optional[Any]:
        cur = obj
        for key in keys:
            if cur is None:
                return None
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                cur = ParaformerBatchASRNode._safe_get_attr(cur, key)
        return cur

    @staticmethod
    def _safe_get_attr(obj: Any, name: str) -> Optional[Any]:
        try:
            return getattr(obj, name)
        except Exception:
            return None

    @staticmethod
    def _to_mapping(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj

        try:
            to_dict_fn = getattr(obj, "to_dict")
            if callable(to_dict_fn):
                converted = to_dict_fn()
                if isinstance(converted, dict):
                    return converted
        except Exception:
            pass

        try:
            data = vars(obj)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        return {}

    @staticmethod
    def _download_transcription_payload(url: str) -> Dict[str, Any]:
        try:
            with urlopen(url, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except (URLError, TimeoutError, ValueError, OSError):
            return {}
        return {}

    @staticmethod
    def _extract_text_from_transcript_payload(payload: Dict[str, Any]) -> str:
        # Success detail example: {"transcripts": [{"text": "..."}]}
        transcripts = payload.get("transcripts")
        if isinstance(transcripts, list) and transcripts:
            lines: List[str] = []
            for transcript in transcripts:
                if not isinstance(transcript, dict):
                    continue
                if transcript.get("text"):
                    lines.append(str(transcript["text"]))
                    continue
                sentences = transcript.get("sentences")
                if isinstance(sentences, list):
                    sentence_text = "".join(
                        str(s.get("text", "")) for s in sentences if isinstance(s, dict)
                    )
                    if sentence_text:
                        lines.append(sentence_text)
            if lines:
                return "\n".join(lines)

        # Some responses may contain top-level sentence list.
        sentence_list = payload.get("sentence_list") or payload.get("sentences")
        if isinstance(sentence_list, list):
            return "".join(
                str(sentence.get("text", ""))
                for sentence in sentence_list
                if isinstance(sentence, dict)
            )

        return str(payload.get("text", "") or "")


NODE_CLASS_MAPPINGS = {
    "ParaformerBatchASRNode": ParaformerBatchASRNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParaformerBatchASRNode": "Bailian Paraformer Batch ASR",
}
