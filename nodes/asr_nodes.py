from __future__ import annotations

import json
import os
import time
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple


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
                        "placeholder": "one OSS object key per line, e.g. output/abc.mp3",
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
                "oss_config_json": ("STRING", {"default": ""}),
                "public_base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "optional, e.g. https://cdn.example.com",
                    },
                ),
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
    ):
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

        raw_items = [line.strip() for line in audio_urls.splitlines() if line.strip()]
        resolved_urls = self._resolve_audio_urls(
            raw_items=raw_items,
            oss_config_json=oss_config_json,
            public_base_url=public_base_url,
        )

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
    def _resolve_audio_urls(
        raw_items: List[str],
        oss_config_json: str,
        public_base_url: str,
    ) -> List[str]:
        if not raw_items:
            return []

        base_url = public_base_url.strip().rstrip("/")
        oss_config: Dict[str, Any] = {}
        if oss_config_json.strip():
            try:
                oss_config = json.loads(oss_config_json)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid oss_config_json") from exc

        resolved: List[str] = []
        for item in raw_items:
            # Backward-compatible path: if a full URL is passed, use it directly.
            if item.startswith("http://") or item.startswith("https://"):
                resolved.append(item)
                continue

            # Primary path: item is an OSS object key.
            key = item.lstrip("/")
            if base_url:
                resolved.append(f"{base_url}/{key}")
                continue

            endpoint = str(oss_config.get("endpoint", "")).rstrip("/")
            bucket = str(oss_config.get("bucket_name", "")).strip()
            if endpoint and bucket:
                resolved.append(f"{endpoint}/{bucket}/{key}")
                continue

            raise ValueError(
                f"Input key '{item}' cannot be resolved to a public URL. "
                "Please provide public_base_url, or pass valid oss_config_json(endpoint + bucket_name)."
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
        output = getattr(response, "output", None)
        if output is None:
            return "", ""

        if hasattr(output, "to_dict"):
            payload = output.to_dict()
        elif isinstance(output, dict):
            payload = output
        else:
            payload = vars(output)

        text = str(payload.get("text", "") or "")
        url = str(
            payload.get("transcription_url")
            or payload.get("url")
            or payload.get("result_url")
            or ""
        )

        if not text:
            results = payload.get("results") or payload.get("result")
            if isinstance(results, list) and results:
                first = results[0] if isinstance(results[0], dict) else {}
                text = str(first.get("text", "") or "")
                url = url or str(
                    first.get("transcription_url")
                    or first.get("url")
                    or first.get("result_url")
                    or ""
                )

        if not text:
            sentence_list = payload.get("sentence_list") or payload.get("sentences")
            if isinstance(sentence_list, list):
                parts = []
                for sentence in sentence_list:
                    if isinstance(sentence, dict) and sentence.get("text"):
                        parts.append(str(sentence["text"]))
                text = "".join(parts)

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
                cur = getattr(cur, key, None)
        return cur


NODE_CLASS_MAPPINGS = {
    "ParaformerBatchASRNode": ParaformerBatchASRNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParaformerBatchASRNode": "Bailian Paraformer Batch ASR",
}
