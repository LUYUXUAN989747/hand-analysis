#!/usr/bin/env python3
"""为切分后的视频片段生成中文场景、动作和手部描述。

这是一个轻量级本地标注阶段：脚本会从每个片段均匀抽取若干帧，
用 MediaPipe 手部关键点估计手势和运动，用 OpenCV 画面统计量生成
保守的场景描述。由于没有额外物体识别模型，脚本不会凭空编造无法
由当前检测结果验证的具体物体名称。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


FINGER_CHAINS = {
    "拇指": (1, 2, 3, 4),
    "食指": (5, 6, 7, 8),
    "中指": (9, 10, 11, 12),
    "无名指": (13, 14, 15, 16),
    "小指": (17, 18, 19, 20),
}


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算以 B 为顶点的 ABC 夹角，单位为度。

    手指是否伸直主要依赖关节夹角，因此这里把三个关键点转换成两条向量，
    再用余弦公式得到夹角。极短向量会导致数值不稳定，直接返回 0。
    """
    ba, bc = a - b, c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def hand_pose(landmarks: list) -> tuple[str, list[str]]:
    points = np.asarray([[item.x, item.y, item.z] for item in landmarks], dtype=np.float32)
    extended: list[str] = []
    for name, chain in FINGER_CHAINS.items():
        mcp, pip, dip, tip = chain
        straight = angle(points[mcp], points[pip], points[dip])
        tip_farther = np.linalg.norm(points[tip] - points[0]) > np.linalg.norm(
            points[pip] - points[0]
        )
        if straight > (145 if name != "拇指" else 130) and tip_farther:
            extended.append(name)

    palm_size = max(
        float(np.linalg.norm(points[0] - points[9])),
        float(np.linalg.norm(points[5] - points[17])),
        1e-5,
    )
    pinch_ratio = float(np.linalg.norm(points[4] - points[8]) / palm_size)
    non_thumb = set(extended) - {"拇指"}
    if pinch_ratio < 0.42:
        pose = "拇指与食指捏合"
    elif len(non_thumb) >= 4:
        pose = "手掌张开"
    elif not non_thumb:
        pose = "握拳或抓握"
    elif non_thumb == {"食指"}:
        pose = "食指伸出"
    elif non_thumb == {"食指", "中指"}:
        pose = "食指和中指伸出"
    elif len(non_thumb) == 3:
        pose = "三指伸展"
    else:
        pose = "半张开或操作姿态"
    return pose, extended


def scene_features(frame: np.ndarray) -> dict:
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    edge_density = float((cv2.Canny(gray, 80, 160) > 0).mean())
    return {
        "brightness": float(mean_hsv[2]),
        "saturation": float(mean_hsv[1]),
        "hue": float(mean_hsv[0]),
        "edge_density": edge_density,
    }


def scene_text(features: list[dict], hand_ratio: float) -> str:
    brightness = float(np.mean([item["brightness"] for item in features]))
    saturation = float(np.mean([item["saturation"] for item in features]))
    edge_density = float(np.mean([item["edge_density"] for item in features]))
    light = "明亮" if brightness >= 165 else "光线适中" if brightness >= 95 else "偏暗"
    color = "色彩较鲜明" if saturation >= 90 else "色彩较柔和"
    detail = "背景物体和纹理较丰富" if edge_density >= 0.12 else "背景结构相对简洁"
    hand = (
        "手持续位于主要操作区域"
        if hand_ratio >= 0.7
        else "手间歇进入主要操作区域"
        if hand_ratio >= 0.3
        else "仅少量采样帧检出手"
    )
    return f"第一视角近距离操作场景，画面{light}、{color}，{detail}；{hand}。"


def motion_metrics(
    observations: list[tuple[float, list[tuple[float, float]]]], duration: float
) -> dict:
    """估计腕部轨迹长度、最大位移和主要运动方向。

    这里使用归一化图像坐标来衡量手在画面中的移动幅度。
    若同一采样帧出现多只手，为了保持简单和稳定，默认取第一条腕部轨迹
    作为该片段的主要运动线索。
    """
    if not observations:
        return {
            "level": "无法判断",
            "large_motion": None,
            "max_displacement": None,
            "path_length": None,
            "direction": "未知",
        }
    primary = [(time, wrists[0]) for time, wrists in observations if wrists]
    if len(primary) < 2:
        return {
            "level": "轻微",
            "large_motion": False,
            "max_displacement": 0.0,
            "path_length": 0.0,
            "direction": "基本静止",
        }
    positions = np.asarray([pos for _, pos in primary], dtype=np.float32)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = float(steps.sum())
    origin_distance = np.linalg.norm(positions - positions[0], axis=1)
    max_displacement = float(origin_distance.max())
    span = positions[-1] - positions[0]
    if abs(float(span[0])) > abs(float(span[1])) * 1.35:
        direction = "整体向右" if span[0] > 0 else "整体向左"
    elif abs(float(span[1])) > abs(float(span[0])) * 1.35:
        direction = "整体向下" if span[1] > 0 else "整体向上"
    else:
        direction = "多方向移动或回到近似原位"

    # 这里的坐标已按图像宽高归一化，所以阈值表示“占画面尺寸的比例”。
    if max_displacement >= 0.35 or path_length >= max(0.75, duration * 0.18):
        level = "大幅"
    elif max_displacement >= 0.15 or path_length >= 0.3:
        level = "中等"
    else:
        level = "轻微"
    return {
        "level": level,
        "large_motion": level == "大幅",
        "max_displacement": round(max_displacement, 4),
        "path_length": round(path_length, 4),
        "direction": direction,
    }


def action_text(hand_ratio: float, motion: dict, dominant_pose: str, pose_changes: int) -> str:
    if hand_ratio <= 0:
        return "采样帧中未稳定检出手，无法可靠判断具体手部动作。"
    if motion["level"] == "大幅":
        verb = "手在画面中大幅移动"
    elif motion["level"] == "中等":
        verb = "手在操作区域内进行中等幅度移动"
    else:
        verb = "手主要在局部区域进行小幅操作或短暂停留"
    change = "，期间手势变化较多" if pose_changes >= 3 else "，手势整体较稳定"
    return f"{verb}，运动趋势为{motion['direction']}；主要呈{dominant_pose}{change}。"


def evenly_spaced_frames(start: int, end: int, count: int) -> list[int]:
    if end <= start:
        return []
    values = np.linspace(start, end - 1, min(count, end - start))
    return sorted(set(int(round(item)) for item in values))


def analyze(args: argparse.Namespace) -> list[dict]:
    metadata_path = Path(args.segments_json).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_path = Path(args.video or metadata["input_video"]).resolve()
    hand_model = Path(args.hand_model).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not hand_model.is_file():
        raise FileNotFoundError(hand_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(metadata["video"]["fps"])
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    results: list[dict] = []

    for number, segment in enumerate(metadata["segments"], 1):
        frame_indices = evenly_spaced_frames(
            int(segment["start_frame"]), int(segment["end_frame"]), args.samples
        )
        visual_features: list[dict] = []
        hand_observations: list[tuple[float, list[tuple[float, float]]]] = []
        poses: list[str] = []
        hand_frames = 0
        max_hands = 0

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            visual_features.append(scene_features(frame))
            if args.detection_width > 0 and frame.shape[1] > args.detection_width:
                scale = args.detection_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (args.detection_width, round(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            landmarks = result.hand_landmarks
            if landmarks:
                hand_frames += 1
                max_hands = max(max_hands, len(landmarks))
                wrists = [(float(hand[0].x), float(hand[0].y)) for hand in landmarks]
                hand_observations.append((frame_idx / fps, wrists))
                poses.extend(hand_pose(hand)[0] for hand in landmarks)

        sample_count = max(1, len(visual_features))
        sampled_hand_ratio = hand_frames / sample_count
        motion = motion_metrics(hand_observations, float(segment["duration_sec"]))
        pose_counts = Counter(poses)
        dominant_pose = pose_counts.most_common(1)[0][0] if poses else "未知姿态"
        pose_changes = sum(a != b for a, b in zip(poses, poses[1:]))
        scene = scene_text(visual_features, sampled_hand_ratio)
        action = action_text(sampled_hand_ratio, motion, dominant_pose, pose_changes)
        description = f"{scene} {action}"
        record = {
            **segment,
            "scene": scene,
            "action": action,
            "description": description,
            "hand": {
                "detected": hand_frames > 0,
                "sampled_hand_ratio": round(sampled_hand_ratio, 4),
                "max_hands": max_hands,
                "large_motion": motion["large_motion"],
                "motion_level": motion["level"],
                "motion_direction": motion["direction"],
                "max_displacement_normalized": motion["max_displacement"],
                "path_length_normalized": motion["path_length"],
                "dominant_pose": dominant_pose,
                "pose_distribution": dict(pose_counts),
                "pose_change_count": pose_changes,
            },
            "annotation_method": "OpenCV scene statistics + MediaPipe hand landmarks + rules",
        }
        results.append(record)
        print(
            f"[{number:02d}/{len(metadata['segments']):02d}] "
            f"segment_{segment['segment_id']:03d}: {description}",
            flush=True,
        )
    cap.release()
    return results


def save_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "segment_descriptions.json"
    json_path.write_text(
        json.dumps(
            {
                "notice": "场景文字仅描述可由画面统计和手部关键点支持的内容，不包含未经检测的物体类别。",
                "segment_count": len(records),
                "segments": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "segment_descriptions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = [
            "segment_id",
            "start_sec",
            "end_sec",
            "duration_sec",
            "scene",
            "action",
            "hand_detected",
            "large_motion",
            "motion_level",
            "dominant_pose",
            "description",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(
                {
                    "segment_id": item["segment_id"],
                    "start_sec": item["start_sec"],
                    "end_sec": item["end_sec"],
                    "duration_sec": item["duration_sec"],
                    "scene": item["scene"],
                    "action": item["action"],
                    "hand_detected": item["hand"]["detected"],
                    "large_motion": item["hand"]["large_motion"],
                    "motion_level": item["hand"]["motion_level"],
                    "dominant_pose": item["hand"]["dominant_pose"],
                    "description": item["description"],
                }
            )

    lines = ["# long1 切分片段语言描述", ""]
    for item in records:
        lines.extend(
            [
                f"## 片段 {item['segment_id']:03d}",
                "",
                f"- 时间：{item['start_sec']:.3f}s - {item['end_sec']:.3f}s",
                f"- 场景：{item['scene']}",
                f"- 动作：{item['action']}",
                f"- 手部大幅移动：{item['hand']['large_motion']}",
                f"- 主要手势：{item['hand']['dominant_pose']}",
                "",
            ]
        )
    (output_dir / "segment_descriptions.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-json", default="output/long1/segments.json")
    parser.add_argument("--video", default=None)
    parser.add_argument("--output-dir", default="output/long1/descriptions")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--detection-width", type=int, default=640)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = analyze(args)
    output_dir = Path(args.output_dir).resolve()
    save_outputs(records, output_dir)
    print(f"Descriptions saved to: {output_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # 避免部分 Windows 主机上 MediaPipe 遥测线程在退出阶段卡住。
    os._exit(0)


if __name__ == "__main__":
    main()
