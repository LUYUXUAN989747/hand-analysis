#!/usr/bin/env python3
"""提取手部姿态几何、掌心朝向和运动轨迹。

默认输入是 long1 以及 `output/long1/segments.json` 中保留下来的片段范围。
脚本会输出逐帧几何 JSONL、轨迹 CSV、统计摘要、二维腕部轨迹图，
以及带有骨架、掌心法向和速度箭头标注的视频，方便后续做动作分析或报告展示。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
COLORS = {"Left": (255, 130, 30), "Right": (40, 210, 80)}


def vector_angle(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))


def point_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return vector_angle(a - b, c - b)


def joint_geometry(points: np.ndarray) -> dict:
    output: dict[str, dict[str, float]] = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        output[name] = {
            "mcp_flexion_deg": round(180.0 - point_angle(points[0], points[mcp], points[pip]), 2),
            "pip_flexion_deg": round(180.0 - point_angle(points[mcp], points[pip], points[dip]), 2),
            "dip_flexion_deg": round(180.0 - point_angle(points[pip], points[dip], points[tip]), 2),
        }
    return output


def classify_pose(points: np.ndarray) -> tuple[str, list[str]]:
    extended: list[str] = []
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        straight = point_angle(points[mcp], points[pip], points[dip])
        farther = np.linalg.norm(points[tip] - points[0]) > np.linalg.norm(
            points[pip] - points[0]
        )
        if straight > (130 if name == "thumb" else 145) and farther:
            extended.append(name)
    palm = max(
        float(np.linalg.norm(points[0] - points[9])),
        float(np.linalg.norm(points[5] - points[17])),
        1e-6,
    )
    pinch = float(np.linalg.norm(points[4] - points[8]) / palm)
    fingers = set(extended) - {"thumb"}
    if pinch < 0.42:
        pose = "pinch"
    elif len(fingers) >= 4:
        pose = "open_palm"
    elif not fingers:
        pose = "fist_or_grasp"
    elif fingers == {"index"}:
        pose = "pointing"
    elif fingers == {"index", "middle"}:
        pose = "two_finger"
    else:
        pose = "manipulation"
    return pose, extended


def palm_geometry(points: np.ndarray, handedness: str) -> dict:
    wrist, index_mcp, pinky_mcp = points[0], points[5], points[17]
    center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    if handedness == "Left":
        normal = -normal
    norm = float(np.linalg.norm(normal))
    if norm > 1e-8:
        normal /= norm
    # MediaPipe 的图像/世界坐标中，z 越负通常表示越靠近相机。
    if normal[2] < -0.35:
        facing = "palm_toward_camera"
    elif normal[2] > 0.35:
        facing = "palm_away_from_camera"
    else:
        facing = "palm_sideways"
    horizontal = "tilted_right" if normal[0] > 0.35 else "tilted_left" if normal[0] < -0.35 else "centered"
    vertical = "tilted_down" if normal[1] > 0.35 else "tilted_up" if normal[1] < -0.35 else "level"
    return {
        "center": [round(float(v), 6) for v in center],
        "normal": [round(float(v), 6) for v in normal],
        "facing": facing,
        "horizontal_tilt": horizontal,
        "vertical_tilt": vertical,
    }


def draw_hand(
    frame: np.ndarray,
    points: np.ndarray,
    handedness: str,
    pose: str,
    palm: dict,
    velocity: np.ndarray,
) -> None:
    height, width = frame.shape[:2]
    pixels = np.column_stack((points[:, 0] * width, points[:, 1] * height)).astype(int)
    color = COLORS.get(handedness, (0, 220, 220))
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), color, 3, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(frame, tuple(point), 4, (245, 245, 245), -1, cv2.LINE_AA)
    center = np.mean(points[[0, 5, 9, 13, 17], :2], axis=0)
    normal = np.asarray(palm["normal"][:2])
    start = (int(center[0] * width), int(center[1] * height))
    end = (
        int((center[0] + normal[0] * 0.13) * width),
        int((center[1] + normal[1] * 0.13) * height),
    )
    cv2.arrowedLine(frame, start, end, (40, 50, 245), 3, cv2.LINE_AA, tipLength=0.25)
    wrist = pixels[0]
    velocity_end = (
        int(wrist[0] + velocity[0] * width * 0.15),
        int(wrist[1] + velocity[1] * height * 0.15),
    )
    cv2.arrowedLine(
        frame, tuple(wrist), velocity_end, (255, 230, 30), 3, cv2.LINE_AA, tipLength=0.25
    )
    label = f"{handedness} | {pose} | {palm['facing']}"
    cv2.putText(
        frame, label, (max(5, wrist[0] - 60), max(25, wrist[1] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )


def retained_frames(metadata: dict, stride: int) -> list[tuple[int, int]]:
    frames: dict[int, int] = {}
    for segment in metadata["segments"]:
        segment_id = int(segment["segment_id"])
        for frame_idx in range(
            int(segment["start_frame"]), int(segment["end_frame"]), stride
        ):
            frames.setdefault(frame_idx, segment_id)
    return sorted(frames.items())


def save_trajectory_image(path: Path, tracks: dict[str, list[dict]]) -> None:
    canvas = np.full((800, 1200, 3), 248, np.uint8)
    cv2.rectangle(canvas, (70, 70), (1130, 730), (80, 80, 80), 2)
    cv2.putText(canvas, "Normalized wrist trajectories", (70, 42), 0, 0.8, (40, 40, 40), 2)
    for handedness, records in tracks.items():
        color = COLORS.get(handedness, (0, 150, 220))
        points = [
            (70 + int(item["wrist_x"] * 1060), 70 + int(item["wrist_y"] * 660))
            for item in records
        ]
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
        if points:
            cv2.circle(canvas, points[0], 8, color, -1)
            cv2.putText(canvas, f"{handedness} start", points[0], 0, 0.55, color, 2)
    cv2.imwrite(str(path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="dataset/long1.mp4")
    parser.add_argument("--segments-json", default="output/long1/segments.json")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--output-dir", default="output/long1/hand_geometry")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--detection-width", type=int, default=640)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    metadata = json.loads(Path(args.segments_json).read_text(encoding="utf-8"))
    model_path = Path(args.hand_model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = float(metadata["video"]["fps"])
    stride = max(1, round(fps / args.sample_fps))
    targets = retained_frames(metadata, stride)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.min_confidence,
        min_hand_presence_confidence=args.min_confidence,
        min_tracking_confidence=0.5,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(output_dir / "hand_geometry.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.sample_fps,
            (width, height),
        )
    previous: dict[str, tuple[float, np.ndarray]] = {}
    tracks: dict[str, list[dict]] = defaultdict(list)
    frame_records: list[dict] = []
    jsonl = (output_dir / "frame_geometry.jsonl").open("w", encoding="utf-8")

    for number, (frame_idx, segment_id) in enumerate(targets, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, full_frame = cap.read()
        if not ok:
            continue
        detect_frame = full_frame
        if args.detection_width > 0 and width > args.detection_width:
            scale = args.detection_width / width
            detect_frame = cv2.resize(
                full_frame,
                (args.detection_width, round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = round(frame_idx * 1000.0 / fps)
        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        timestamp_sec = frame_idx / fps
        hands: list[dict] = []
        # 每个 handedness 只保留置信度最高的一只手。MediaPipe 偶尔会把两只手
        # 都标成同一个 Left/Right，如果直接都写入轨迹，会造成同一时间戳下的
        # 轨迹跳变，速度计算也会出现不合理的零时间间隔。
        best_index: dict[str, int] = {}
        for candidate_index, categories in enumerate(result.handedness):
            label = categories[0].category_name
            old_index = best_index.get(label)
            if (
                old_index is None
                or categories[0].score > result.handedness[old_index][0].score
            ):
                best_index[label] = candidate_index
        for index in best_index.values():
            image_lm = result.hand_landmarks[index]
            world_lm = result.hand_world_landmarks[index]
            category = result.handedness[index][0]
            handedness = category.category_name
            image_points = np.asarray(
                [[item.x, item.y, item.z] for item in image_lm], dtype=np.float32
            )
            world_points = np.asarray(
                [[item.x, item.y, item.z] for item in world_lm], dtype=np.float32
            )
            pose, extended = classify_pose(image_points)
            palm = palm_geometry(world_points, handedness)
            wrist = image_points[0]
            if handedness in previous:
                old_time, old_wrist = previous[handedness]
                dt = timestamp_sec - old_time
                velocity = (
                    (wrist - old_wrist) / dt
                    if dt > 1e-5
                    else np.zeros(3, dtype=np.float32)
                )
            else:
                velocity = np.zeros(3, dtype=np.float32)
            previous[handedness] = (timestamp_sec, wrist.copy())
            speed = float(np.linalg.norm(velocity[:2]))
            geometry = {
                "handedness": handedness,
                "handedness_score": round(float(category.score), 5),
                "pose": pose,
                "extended_fingers": extended,
                "palm": palm,
                "joint_angles": joint_geometry(world_points),
                "wrist_velocity_normalized_per_sec": [
                    round(float(value), 6) for value in velocity
                ],
                "wrist_speed_normalized_per_sec": round(speed, 6),
                "landmarks_image_normalized": [
                    [round(float(v), 6) for v in point] for point in image_points
                ],
                "landmarks_world_meter": [
                    [round(float(v), 6) for v in point] for point in world_points
                ],
            }
            hands.append(geometry)
            tracks[handedness].append(
                {
                    "frame": frame_idx,
                    "time_sec": timestamp_sec,
                    "segment_id": segment_id,
                    "wrist_x": float(wrist[0]),
                    "wrist_y": float(wrist[1]),
                    "wrist_z": float(wrist[2]),
                    "speed": speed,
                    "pose": pose,
                    "palm_facing": palm["facing"],
                }
            )
            if writer is not None:
                draw_hand(full_frame, image_points, handedness, pose, palm, velocity)
        record = {
            "frame": frame_idx,
            "time_sec": round(timestamp_sec, 6),
            "segment_id": segment_id,
            "hands": hands,
        }
        frame_records.append(record)
        jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        if writer is not None:
            cv2.putText(
                full_frame,
                f"segment {segment_id:03d} | t={timestamp_sec:.2f}s | hands={len(hands)}",
                (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 230, 240), 2, cv2.LINE_AA,
            )
            writer.write(full_frame)
        if number % 100 == 0:
            print(f"\rGeometry: {number}/{len(targets)} sampled frames", end="", flush=True)

    jsonl.close()
    cap.release()
    if writer is not None:
        writer.release()

    with (output_dir / "trajectories.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = [
            "handedness", "frame", "time_sec", "segment_id", "wrist_x", "wrist_y",
            "wrist_z", "speed", "pose", "palm_facing",
        ]
        csv_writer = csv.DictWriter(file, fieldnames=fields)
        csv_writer.writeheader()
        for handedness, records in tracks.items():
            for item in records:
                csv_writer.writerow({"handedness": handedness, **item})

    summaries = {}
    for handedness, records in tracks.items():
        positions = np.asarray(
            [[item["wrist_x"], item["wrist_y"], item["wrist_z"]] for item in records]
        )
        path_length = (
            float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            if len(positions) > 1 else 0.0
        )
        summaries[handedness] = {
            "observation_count": len(records),
            "path_length_normalized": round(path_length, 5),
            "mean_speed_normalized_per_sec": round(
                float(np.mean([item["speed"] for item in records])), 5
            ),
            "max_speed_normalized_per_sec": round(
                float(max(item["speed"] for item in records)), 5
            ),
            "pose_distribution": dict(Counter(item["pose"] for item in records)),
            "palm_orientation_distribution": dict(
                Counter(item["palm_facing"] for item in records)
            ),
        }
    (output_dir / "geometry_summary.json").write_text(
        json.dumps(
            {
                "input_video": str(video_path),
                "sample_fps": args.sample_fps,
                "coordinate_notes": {
                    "image": "x/y normalized to frame; z is MediaPipe relative depth",
                    "world": "MediaPipe metric-scale hand-local 3-D coordinates in meters",
                    "palm_normal": "unit normal derived from wrist/index-MCP/pinky-MCP",
                    "velocity": "normalized image coordinates per second",
                },
                "hands": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_trajectory_image(output_dir / "wrist_trajectories.png", tracks)
    print(f"\nGeometry outputs saved to: {output_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
