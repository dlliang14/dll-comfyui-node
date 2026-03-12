# ComfyUI Custom Nodes - Audio Pipeline

这是一个用于 ComfyUI 的自定义节点库（起步版）。

当前已实现节点：
- `FFmpeg Batch Convert (Video -> Audio)`

## 功能
- 支持 `directory / list / single` 三种输入模式
- 支持批量把视频转换为 `mp3` 或 `wav`
- 支持覆盖策略：`skip / overwrite / rename`
- 默认使用 `imageio-ffmpeg` 自动获取 ffmpeg，可免系统预装

## 安装
1. 将仓库放到 ComfyUI 的 `custom_nodes` 目录下。
2. 安装依赖：
   - `pip install -r requirements.txt`
3. 重启 ComfyUI。

## 节点说明
### FFmpeg Batch Convert (Video -> Audio)
- `input_mode`:
  - `directory`: 使用 `source_path` + `glob_pattern` 扫描文件
  - `list`: 使用 `file_list`（每行一个绝对路径）
  - `single`: 使用 `source_path` 单文件
- `output_format`: `mp3` / `wav`
- `sample_rate`: 采样率，默认 `44100`
- `channels`: 声道数，默认 `2`
- `audio_bitrate`: 仅在 `mp3` 生效，默认 `192k`
- `overwrite`: `skip / overwrite / rename`
- `continue_on_error`: 出错后是否继续处理后续文件
- `ffmpeg_path`: 可选高级参数，留空时自动发现 ffmpeg

## 输出
- `output_files`: 成功输出文件路径（按行拼接）
- `success_count`: 成功数量
- `fail_count`: 失败数量
- `report_json`: 结构化 JSON 报告

## Linux Pod + Windows UI 场景建议
- UI 中路径请使用容器内路径，例如 `/data/input`、`/data/output`
- 不要传 Windows 盘符路径
- 如集群无公网，建议在镜像构建阶段预热 ffmpeg 或设置 `IMAGEIO_FFMPEG_EXE`
