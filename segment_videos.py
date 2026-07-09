#!/usr/bin/env python3
"""按视觉环境和手部出现情况切分第一视角视频。

整体流程分为两个时间层级：
1. 先用 HSV 颜色直方图距离寻找粗粒度的环境/镜头变化边界；
2. 再在每个粗场景内部用 MediaPipe 检测手部，只保留手稳定出现的片段。

脚本使用 OpenCV 负责视频读取和导出，使用 MediaPipe 负责手部检测。
输出的 JSON、Numpy 和视频片段会尽量保持机器可读，方便后续的动作描述、
手部几何分析或机器人重定向流程继续复用同一份片段元数据。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_OPEN_TASKS: list[object] = []


@dataclass
class Segment:
    segment_id: int
    scene_id: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    hand_ratio: float
    clip_path: str


def frame_histogram(frame: np.ndarray) -> np.ndarray:
    """计算归一化 HSV 直方图，用于衡量两帧画面环境是否发生明显变化。

    先把帧缩小到固定尺寸，能降低噪声和运行成本；再使用 H/S 通道而不是
    原始 RGB，这样对轻微亮度变化更稳健，不会因为曝光波动就误判为换场景。
    """
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def detect_scene_boundaries(
    video_path: Path,
    fps: float,
    frame_count: int,
    sample_fps: float,
    threshold: float | None,
    min_scene_sec: float,
) -> tuple[list[int], dict]:
    """根据相邻采样帧的直方图跳变寻找粗场景边界。

    如果用户没有手动给定阈值，就使用距离分布的高分位数自适应估计。
    同时用 `min_scene_sec` 限制相邻边界的最小间隔，避免第一视角晃动时
    把同一个连续环境切得过碎。
    """
    cap = cv2.VideoCapture(str(video_path))
    stride = max(1, round(fps / sample_fps))
    sampled_frames: list[int] = []
    distances: list[float] = []
    previous = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            hist = frame_histogram(frame)
            if previous is not None:
                distances.append(float(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA)))
                sampled_frames.append(frame_idx)
            previous = hist
        frame_idx += 1
    cap.release()

    if not distances:
        return [0, frame_count], {"threshold": None, "samples": 0, "candidates": 0}

    values = np.asarray(distances)
    percentile_99 = float(np.percentile(values, 99))
    if threshold is None:
        # 第一视角视频经常有手持晃动和近距离遮挡，中等幅度的直方图变化很多。
        # 若使用 MAD 之类的稳健离群规则，阈值有时会高到超过实际观测范围。
        # 因此这里取最高 1% 距离尾部作为“异常环境变化”的自适应定义。
        threshold = max(0.32, percentile_99)

    min_gap = max(1, round(min_scene_sec * fps))
    candidates = [
        (sampled_frames[i], distances[i])
        for i in range(len(distances))
        if distances[i] >= threshold
        and distances[i] >= max(distances[max(0, i - 1) : min(len(distances), i + 2)])
    ]
    boundaries = [0]
    for frame_idx, _score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(frame_idx - existing) >= min_gap for existing in boundaries):
            boundaries.append(frame_idx)
    boundaries.append(frame_count)
    boundaries.sort()
    return boundaries, {
        "threshold": round(float(threshold), 5),
        "distance_p99": round(percentile_99, 5),
        "distance_max": round(float(values.max()), 5),
        "samples": len(distances),
        "candidates": len(candidates),
    }


def detect_hands(
    video_path: Path,
    frame_count: int,
    fps: float,
    hand_model: Path,
    min_detection_confidence: float,
    sample_fps: float,
    detection_width: int,
    checkpoint_path: Path | None = None,
    exit_after_checkpoint: bool = False,
) -> np.ndarray:
    """生成逐帧手部存在掩码。

    为了节省时间，实际只按 `sample_fps` 抽样跑 MediaPipe；随后把每个抽样
    判断扩展到它附近的时间单元。输出数组长度与视频总帧数一致，其中 True
    表示该帧所在时间附近检测到至少一只手。
    """
    present = np.zeros(frame_count, dtype=bool)
    cap = cv2.VideoCapture(str(video_path))
    stride = max(1, round(fps / sample_fps))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    # 保留 detector 引用直到进程强制退出。部分 Windows 版 MediaPipe Tasks
    # 在遥测清理阶段可能因为本地时钟/网络状态卡住；延长对象生命周期并在
    # main 结尾 os._exit(0)，可以避免脚本已经完成却迟迟不退出。
    _OPEN_TASKS.append(detector)
    idx = 0
    sampled_indices: list[int] = []
    sampled_values: list[bool] = []
    while idx < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            if detection_width > 0 and frame.shape[1] > detection_width:
                scale = detection_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (detection_width, round(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = round(idx * 1000.0 / fps)
            sampled_indices.append(idx)
            sampled_values.append(
                bool(detector.detect_for_video(image, timestamp_ms).hand_landmarks)
            )
        idx += 1
        if idx % 500 == 0:
            print(f"\rHand detection: {idx}/{frame_count} frames", end="", flush=True)
    # 把抽样帧的检测结果扩展到离它最近的一段连续帧，形成逐帧掩码。
    for sample_no, sample_idx in enumerate(sampled_indices):
        start = 0 if sample_no == 0 else (sampled_indices[sample_no - 1] + sample_idx) // 2
        end = (
            frame_count
            if sample_no == len(sampled_indices) - 1
            else (sample_idx + sampled_indices[sample_no + 1]) // 2
        )
        present[start:end] = sampled_values[sample_no]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(checkpoint_path, present)
        print(f"\nSaved hand mask checkpoint: {checkpoint_path}", flush=True)
        if exit_after_checkpoint:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    cap.release()
    print(f"\rHand detection: {int(present.size)}/{frame_count} frames")
    return present


def bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """填补被前后手部检出夹住的短暂漏检。

    手部检测偶尔会在一两帧里丢失，直接切段会导致一个动作被打碎。
    只要漏检区间足够短，并且前后都是 True，就把这段补成 True。
    """
    result = mask.copy()
    false_indices = np.flatnonzero(~result)
    if not false_indices.size:
        return result
    runs = np.split(false_indices, np.where(np.diff(false_indices) != 1)[0] + 1)
    for run in runs:
        start, end = int(run[0]), int(run[-1])
        if len(run) <= max_gap and start > 0 and end + 1 < len(result):
            if result[start - 1] and result[end + 1]:
                result[start : end + 1] = True
    return result


def build_segments(
    scene_boundaries: list[int],
    raw_hand_mask: np.ndarray,
    fps: float,
    max_gap_sec: float,
    min_hand_sec: float,
    padding_sec: float,
    min_hand_ratio: float,
) -> list[tuple[int, int, int, float]]:
    """在每个粗场景内生成包含手部的有效子片段。

    片段不会跨越场景边界；每段至少需要达到 `min_hand_sec`，
    且原始手部检出比例需要高于 `min_hand_ratio`。两侧 padding 用于
    保留动作开始前和结束后的少量上下文。
    """
    segments: list[tuple[int, int, int, float]] = []
    max_gap = round(max_gap_sec * fps)
    min_frames = max(1, round(min_hand_sec * fps))
    padding = round(padding_sec * fps)

    for scene_id, (scene_start, scene_end) in enumerate(
        zip(scene_boundaries[:-1], scene_boundaries[1:])
    ):
        raw = raw_hand_mask[scene_start:scene_end]
        stable = bridge_short_gaps(raw, max_gap)
        indices = np.flatnonzero(stable)
        if not indices.size:
            continue
        runs = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        for run in runs:
            if len(run) < min_frames:
                continue
            start = max(scene_start, scene_start + int(run[0]) - padding)
            end = min(scene_end, scene_start + int(run[-1]) + 1 + padding)
            ratio = float(raw_hand_mask[start:end].mean())
            if ratio >= min_hand_ratio:
                segments.append((scene_id, start, end, ratio))
    return segments


def export_outputs(
    video_path: Path,
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
    segments_data: list[tuple[int, int, int, float]],
) -> list[Segment]:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old_clip in clips_dir.glob("segment_*.mp4"):
        old_clip.unlink()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    records: list[Segment] = []
    combined_path = output_dir / f"{video_path.stem}_hand_segments.mp4"
    combined = cv2.VideoWriter(str(combined_path), fourcc, fps, (width, height))
    if not combined.isOpened():
        raise RuntimeError("Cannot create output video; check OpenCV codec support.")

    for segment_id, (scene_id, start, end, hand_ratio) in enumerate(segments_data):
        clip_path = clips_dir / f"segment_{segment_id:03d}.mp4"
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            combined.write(frame)
        cap.release()
        writer.release()
        records.append(
            Segment(
                segment_id=segment_id,
                scene_id=scene_id,
                start_frame=start,
                end_frame=end,
                start_sec=round(start / fps, 3),
                end_sec=round(end / fps, 3),
                duration_sec=round((end - start) / fps, 3),
                hand_ratio=round(hand_ratio, 4),
                clip_path=str(clip_path.relative_to(output_dir)),
            )
        )
    combined.release()
    return records


def save_timeline(
    path: Path,
    duration_sec: float,
    boundaries: list[int],
    segments: list[Segment],
    fps: float,
) -> None:
    """生成时间轴预览图，直观展示环境边界和最终保留片段。

    蓝色竖线表示粗场景边界，绿色粗线表示被导出的手部片段。
    这里直接用 OpenCV 画图，不依赖额外绘图库，方便在轻量环境下运行。
    """
    width, height = 1600, 220
    canvas = np.full((height, width, 3), 248, np.uint8)
    left, right, y = 80, width - 40, 105
    cv2.line(canvas, (left, y), (right, y), (90, 90, 90), 3)

    def xpos(sec: float) -> int:
        return left + round((right - left) * sec / max(duration_sec, 1e-6))

    for boundary in boundaries[1:-1]:
        x = xpos(boundary / fps)
        cv2.line(canvas, (x, 55), (x, 155), (60, 150, 220), 2)
    for segment in segments:
        cv2.line(
            canvas,
            (xpos(segment.start_sec), y),
            (xpos(segment.end_sec), y),
            (40, 175, 70),
            16,
        )
    cv2.putText(canvas, "blue: environment boundary", (80, 32), 0, 0.65, (60, 150, 220), 2)
    cv2.putText(canvas, "green: retained hand segment", (450, 32), 0, 0.65, (40, 175, 70), 2)
    for sec in np.linspace(0, duration_sec, 6):
        x = xpos(float(sec))
        cv2.line(canvas, (x, y + 25), (x, y + 34), (60, 60, 60), 1)
        cv2.putText(canvas, f"{sec:.1f}s", (x - 24, y + 60), 0, 0.5, (50, 50, 50), 1)
    cv2.imwrite(str(path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default="dataset/long1.mp4")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--scene-sample-fps", type=float, default=2.0)
    parser.add_argument("--scene-threshold", type=float, default=None)
    parser.add_argument("--min-scene-sec", type=float, default=2.0)
    parser.add_argument("--max-hand-gap-sec", type=float, default=0.35)
    parser.add_argument("--min-hand-sec", type=float, default=0.7)
    parser.add_argument("--padding-sec", type=float, default=0.2)
    parser.add_argument("--min-hand-ratio", type=float, default=0.35)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--hand-sample-fps", type=float, default=10.0)
    parser.add_argument("--hand-detection-width", type=int, default=640)
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="只检测并保存手部掩码，不导出片段；通常只在调试或大视频复跑时使用。",
    )
    parser.add_argument(
        "--reuse-hand-mask",
        action="store_true",
        help="复用之前显式保存的 hand_presence.npy，跳过 MediaPipe 手部检测。",
    )
    parser.add_argument(
        "--save-hand-mask",
        action="store_true",
        help="保留兼容参数；脚本现在默认保存 hand_presence.npy，便于复跑和诊断。",
    )
    parser.add_argument("--clean", action="store_true", help="Delete the old output directory first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    output_dir = Path(args.output_dir or f"output/{video_path.stem}").resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    hand_model = Path(args.hand_model).resolve()
    if not hand_model.is_file():
        raise FileNotFoundError(f"MediaPipe hand model not found: {hand_model}")
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise RuntimeError("Invalid FPS or frame count reported by decoder.")
    print(f"Input: {video_path.name} | {frame_count} frames | {fps:.3f} FPS")

    boundaries, scene_stats = detect_scene_boundaries(
        video_path,
        fps,
        frame_count,
        args.scene_sample_fps,
        args.scene_threshold,
        args.min_scene_sec,
    )
    print(f"Coarse environments: {len(boundaries) - 1} | {scene_stats}")
    hand_mask_path = output_dir / "hand_presence.npy"
    if args.reuse_hand_mask:
        raw_hand_mask = np.load(hand_mask_path)
        if raw_hand_mask.shape != (frame_count,):
            raise ValueError(f"Hand mask shape mismatch: {raw_hand_mask.shape}")
        print(f"Reusing hand mask: {hand_mask_path}")
    else:
        raw_hand_mask = detect_hands(
            video_path,
            frame_count,
            fps,
            hand_model,
            args.min_detection_confidence,
            args.hand_sample_fps,
            args.hand_detection_width,
            checkpoint_path=hand_mask_path,
            exit_after_checkpoint=args.detect_only,
        )
    segments_data = build_segments(
        boundaries,
        raw_hand_mask,
        fps,
        args.max_hand_gap_sec,
        args.min_hand_sec,
        args.padding_sec,
        args.min_hand_ratio,
    )
    records = export_outputs(video_path, output_dir, fps, width, height, segments_data)
    duration = frame_count / fps
    metadata = {
        "input_video": str(video_path),
        "video": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": round(duration, 3),
        },
        "parameters": vars(args),
        "scene_detection": scene_stats,
        "scene_boundaries_sec": [round(frame / fps, 3) for frame in boundaries],
        "summary": {
            "scene_count": len(boundaries) - 1,
            "retained_segment_count": len(records),
            "retained_duration_sec": round(sum(item.duration_sec for item in records), 3),
        },
        "segments": [asdict(item) for item in records],
    }
    with (output_dir / "segments.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    save_timeline(output_dir / "timeline.png", duration, boundaries, records, fps)
    print(
        f"Done: retained {len(records)} segments "
        f"({metadata['summary']['retained_duration_sec']:.2f}s) in {output_dir}"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    # 避免部分 Windows 主机上 MediaPipe Tasks 遥测线程在退出阶段卡住。
    os._exit(0)


if __name__ == "__main__":
    main()
