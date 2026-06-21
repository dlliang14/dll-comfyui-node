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

## Paraformer 转写缓存

`ParaformerBatchASRNode` 会将成功的转写结果保存为 JSON。默认缓存目录是：

```text
/root/ComfyUI/output/asr_cache
```

缓存文件不会被节点自动删除。缓存键由音频 URL（忽略临时签名参数）、模型和语言提示共同生成。

- `use_cache`：优先读取缓存；未命中时调用 Paraformer，并保存结果。
- `refresh`：忽略已有缓存，重新调用 Paraformer并覆盖对应缓存。
- `cache_only`：只读取缓存，绝不调用 Paraformer；未命中时在报告中返回失败。

缓存命中后，`texts` 输出就是已保存的转写文本，可以直接连接预览或文本保存节点，不必连接 Gemini。`report_json` 会提供 `cache_path`、`cache_hit_count` 和 `asr_request_count`。

## Linux Pod + Windows UI 场景建议
- UI 中路径请使用容器内路径，例如 `/data/input`、`/data/output`
- 不要传 Windows 盘符路径
- 如集群无公网，建议在镜像构建阶段预热 ffmpeg 或设置 `IMAGEIO_FFMPEG_EXE`
