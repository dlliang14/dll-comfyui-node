from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, List

import imageio_ffmpeg


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
                "glob_pattern": ("STRING", {"default": "*.mp4", "multiline": False}),
                "recursive": ("BOOLEAN", {"default": False}),
                "output_dir": (
                    "STRING",
                    {"default": "/root/ComfyUI/output/", "multiline": False},
                ),
                "output_format": (["mp3", "wav"],),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000}),
                "channels": ("INT", {"default": 2, "min": 1, "max": 2}),
                "audio_bitrate": ("STRING", {"default": "192k", "multiline": False}),
                "overwrite": (["skip", "overwrite", "rename"],),
                "continue_on_error": ("BOOLEAN", {"default": True}),
                "ffmpeg_path": ("STRING", {"default": "", "multiline": False}),
            }
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
        output_dir: str,
        output_format: str,
        sample_rate: int,
        channels: int,
        audio_bitrate: str,
        overwrite: str,
        continue_on_error: bool,
        ffmpeg_path: str,
    ):
        ffmpeg_exe = self._resolve_ffmpeg_executable(ffmpeg_path)
        input_files = self._resolve_input_files(
            input_mode=input_mode,
            source_path=source_path,
            file_list=file_list,
            glob_pattern=glob_pattern,
            recursive=recursive,
        )

        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        converted_files: List[str] = []
        success_count = 0
        fail_count = 0
        report = {
            "input_mode": input_mode,
            "output_format": output_format,
            "output_dir": str(output_root),
            "total_inputs": len(input_files),
            "items": [],
        }

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

            dst_path = output_root / f"{src_path.stem}.{output_format}"
            if dst_path.exists():
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
                overwrite=(overwrite == "overwrite"),
            )

            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                success_count += 1
                converted_files.append(str(dst_path))
                report["items"].append(
                    {
                        "input": str(src_path),
                        "output": str(dst_path),
                        "status": "success",
                    }
                )
            else:
                fail_count += 1
                report["items"].append(
                    {
                        "input": str(src_path),
                        "output": str(dst_path),
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


NODE_CLASS_MAPPINGS = {
    "FFmpegBatchConvertNode": FFmpegBatchConvertNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FFmpegBatchConvertNode": "FFmpeg Batch Convert (Video -> Audio)",
}
