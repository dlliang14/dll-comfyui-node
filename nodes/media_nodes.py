from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Iterable, List

import boto3
import imageio_ffmpeg
from botocore.config import Config


class OSSInfoNode:
    """OSS configuration node - provides endpoint, credentials, and bucket info for other nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint": (
                    "STRING",
                    {
                        "default": "https://oss-cn-beijing.aliyuncs.com",
                        "multiline": False,
                    },
                ),
                "access_key_id": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "access_key_secret": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "bucket_name": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "path_prefix": (
                    "STRING",
                    {"default": "comfyui/output/", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("oss_config_json",)
    FUNCTION = "get_config"
    CATEGORY = "dlliang14/oss"

    def get_config(
        self,
        endpoint: str,
        access_key_id: str,
        access_key_secret: str,
        bucket_name: str,
        path_prefix: str,
    ):
        config = {
            "endpoint": endpoint.strip(),
            "access_key_id": access_key_id.strip(),
            "access_key_secret": access_key_secret.strip(),
            "bucket_name": bucket_name.strip(),
            "path_prefix": path_prefix.rstrip("/") + "/",
        }
        return (json.dumps(config, ensure_ascii=False),)


class FFmpegBatchConvertNode:
    """Batch convert videos to audio files using ffmpeg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mode": (["directory", "list", "single"],),
                "source_path": (
                    "STRING",
                    {"default": "/root/ComfyUI/input/", "multiline": False},
                ),
                "file_list": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "list mode: one absolute path per line",
                    },
                ),
                "glob_pattern": (
                    "STRING",
                    {"default": "*.mp4", "multiline": False},
                ),
                "recursive": ("BOOLEAN", {"default": False}),
                "output_format": (["mp3", "wav"],),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000}),
                "channels": ("INT", {"default": 2, "min": 1, "max": 2}),
                "audio_bitrate": ("STRING", {"default": "192k", "multiline": False}),
                "overwrite": (["skip", "overwrite", "rename"],),
                "continue_on_error": ("BOOLEAN", {"default": True}),
                "ffmpeg_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "oss_config_json": ("STRING", {"default": ""}),
                "use_local_output": ("BOOLEAN", {"default": True}),
                "local_output_dir": (
                    "STRING",
                    {"default": "/root/ComfyUI/output/", "multiline": False},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("output_files", "success_count", "fail_count", "report_json")
    FUNCTION = "convert"
    CATEGORY = "dlliang14/media"

    def convert(
        self,
        input_mode: str,
        source_path: str,
        file_list: str,
        glob_pattern: str,
        recursive: bool,
        output_format: str,
        sample_rate: int,
        channels: int,
        audio_bitrate: str,
        overwrite: str,
        continue_on_error: bool,
        ffmpeg_path: str,
        oss_config_json: str = "",
        use_local_output: bool = True,
        local_output_dir: str = "/root/ComfyUI/output/",
    ):
        ffmpeg_exe = self._resolve_ffmpeg_executable(ffmpeg_path)
        input_files = self._resolve_input_files(
            input_mode=input_mode,
            source_path=source_path,
            file_list=file_list,
            glob_pattern=glob_pattern,
            recursive=recursive,
        )

        # Parse OSS config if provided
        oss_config = None
        if oss_config_json and oss_config_json.strip():
            try:
                oss_config = json.loads(oss_config_json)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid OSS config JSON: {oss_config_json}")

        # Determine output directory
        if use_local_output and not oss_config:
            output_root = Path(local_output_dir).expanduser().resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            use_oss = False
        elif oss_config:
            use_oss = True
            output_root = None  # OSS paths will be generated on the fly
        else:
            output_root = Path(local_output_dir).expanduser().resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            use_oss = False

        converted_files: List[str] = []
        success_count = 0
        fail_count = 0
        report = {
            "input_mode": input_mode,
            "output_format": output_format,
            "use_oss": use_oss,
            "total_inputs": len(input_files),
            "items": [],
        }

        if use_oss:
            assert oss_config is not None
            report["oss_bucket"] = oss_config.get("bucket_name")
            report["oss_prefix"] = oss_config.get("path_prefix")
        else:
            assert output_root is not None
            report["output_dir"] = str(output_root)

        for src in input_files:
            src_path = Path(src).expanduser().resolve()
            if not src_path.exists() or not src_path.is_file():
                fail_count += 1
                report["items"].append(
                    {
                        "input": str(src_path),
                        "status": "failed",
                        "error": "input file not found",
                    }
                )
                if not continue_on_error:
                    raise RuntimeError(f"Input file not found: {src_path}")
                continue

            # Generate output filename
            if use_oss:
                # OSS path: bucket://prefix/uuid.format
                assert oss_config is not None
                file_uuid = str(uuid.uuid4())
                dst_filename = f"{file_uuid}.{output_format}"
                dst_oss_path = f"{oss_config['path_prefix']}{dst_filename}"
                # Local temp file for ffmpeg output
                dst_path = Path(local_output_dir).expanduser() / dst_filename
                dst_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                assert output_root is not None
                dst_path = output_root / f"{src_path.stem}.{output_format}"
                dst_oss_path = None

            if not use_oss and dst_path.exists():
                if overwrite == "skip":
                    report["items"].append(
                        {
                            "input": str(src_path),
                            "output": str(dst_path),
                            "status": "skipped",
                            "reason": "output exists",
                        }
                    )
                    continue
                if overwrite == "rename":
                    dst_path = self._build_renamed_path(dst_path)

            command = self._build_ffmpeg_command(
                ffmpeg_exe=ffmpeg_exe,
                input_file=src_path,
                output_file=dst_path,
                output_format=output_format,
                sample_rate=sample_rate,
                channels=channels,
                audio_bitrate=audio_bitrate,
                overwrite=(overwrite == "overwrite" or use_oss),
            )

            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                # Upload to OSS if configured
                if use_oss:
                    assert oss_config is not None
                    assert dst_oss_path is not None
                    try:
                        self._upload_to_s3(oss_config, dst_path, dst_oss_path)
                        dst_path.unlink(missing_ok=True)  # remove temp file
                    except Exception as exc:
                        fail_count += 1
                        report["items"].append(
                            {
                                "input": str(src_path),
                                "output": dst_oss_path,
                                "status": "failed",
                                "error": f"OSS upload failed: {exc}",
                            }
                        )
                        if not continue_on_error:
                            raise
                        continue
                success_count += 1
                output_path = dst_oss_path if use_oss else str(dst_path)
                assert output_path is not None
                converted_files.append(output_path)
                report["items"].append(
                    {
                        "input": str(src_path),
                        "output": output_path,
                        "status": "success",
                    }
                )
            else:
                fail_count += 1
                report["items"].append(
                    {
                        "input": str(src_path),
                        "output": dst_oss_path if use_oss else str(dst_path),
                        "status": "failed",
                        "error": (result.stderr or result.stdout).strip(),
                    }
                )
                if not continue_on_error:
                    raise RuntimeError((result.stderr or result.stdout).strip())

        report["success_count"] = success_count
        report["fail_count"] = fail_count

        return (
            "\n".join(converted_files),
            success_count,
            fail_count,
            json.dumps(report, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _resolve_ffmpeg_executable(ffmpeg_path: str) -> str:
        provided = (ffmpeg_path or "").strip()
        if provided:
            path = Path(provided).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"ffmpeg_path does not exist: {path}")
            return str(path)

        env_path = os.getenv("IMAGEIO_FFMPEG_EXE", "").strip()
        if env_path:
            path = Path(env_path).expanduser()
            if path.exists():
                return str(path)

        return imageio_ffmpeg.get_ffmpeg_exe()

    @staticmethod
    def _resolve_input_files(
        input_mode: str,
        source_path: str,
        file_list: str,
        glob_pattern: str,
        recursive: bool,
    ) -> List[str]:
        if input_mode == "single":
            value = (source_path or "").strip()
            return [value] if value else []

        if input_mode == "list":
            return [line.strip() for line in file_list.splitlines() if line.strip()]

        root = Path(source_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            return []

        iterator: Iterable[Path]
        if recursive:
            iterator = root.rglob(glob_pattern)
        else:
            iterator = root.glob(glob_pattern)
        return [str(p.resolve()) for p in iterator if p.is_file()]

    @staticmethod
    def _build_ffmpeg_command(
        ffmpeg_exe: str,
        input_file: Path,
        output_file: Path,
        output_format: str,
        sample_rate: int,
        channels: int,
        audio_bitrate: str,
        overwrite: bool,
    ) -> List[str]:
        command = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(input_file),
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
        ]

        if output_format == "mp3":
            command.extend(["-ab", audio_bitrate, "-f", "mp3"])
        elif output_format == "wav":
            command.extend(["-f", "wav"])
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        command.append(str(output_file))
        return command

    @staticmethod
    def _build_renamed_path(path: Path) -> Path:
        candidate = path
        index = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
            index += 1
        return candidate

    @staticmethod
    def _upload_to_s3(oss_config: dict, local_path: Path, oss_key: str) -> None:
        """Upload a local file to an S3-compatible bucket (Aliyun OSS, Cloudflare R2, AWS S3, etc.).

        oss_config keys:
            endpoint         - full URL, e.g. https://oss-cn-beijing.aliyuncs.com
                               or https://<accountid>.r2.cloudflarestorage.com
            access_key_id    - Access Key ID
            access_key_secret - Access Key Secret
            bucket_name      - target bucket
        """
        endpoint = str(oss_config.get("endpoint", "")).strip()
        access_key_id = str(oss_config.get("access_key_id", "")).strip()
        access_key_secret = str(oss_config.get("access_key_secret", "")).strip()
        bucket_name = str(oss_config.get("bucket_name", "")).strip()
        region_name = str(oss_config.get("region_name", "auto")).strip() or "auto"

        if (
            not endpoint
            or not access_key_id
            or not access_key_secret
            or not bucket_name
        ):
            raise ValueError(
                "Invalid OSS config: endpoint/access_key_id/access_key_secret/bucket_name are required"
            )

        key = oss_key.lstrip("/")
        body = local_path.read_bytes()

        # For Aliyun OSS/S3-compatible services, avoid aws-chunked streaming upload.
        # Use direct put_object with payload signing disabled and checksum policy relaxed.
        base_config = {
            "signature_version": "s3v4",
            "request_checksum_calculation": "when_required",
            "response_checksum_validation": "when_required",
        }

        last_error: Exception | None = None
        for addressing_style in ("virtual", "path"):
            try:
                s3 = boto3.client(
                    "s3",
                    endpoint_url=endpoint,
                    aws_access_key_id=access_key_id,
                    aws_secret_access_key=access_key_secret,
                    region_name=region_name,
                    config=Config(
                        **base_config,
                        s3={
                            "addressing_style": addressing_style,
                            "payload_signing_enabled": False,
                        },
                    ),
                )
                s3.put_object(Bucket=bucket_name, Key=key, Body=body)
                return
            except Exception as exc:  # pragma: no cover
                last_error = exc

        if last_error is not None:
            raise last_error


NODE_CLASS_MAPPINGS = {
    "OSSInfoNode": OSSInfoNode,
    "FFmpegBatchConvertNode": FFmpegBatchConvertNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OSSInfoNode": "OSS Configuration",
    "FFmpegBatchConvertNode": "FFmpeg Batch Convert (Video -> Audio)",
}
