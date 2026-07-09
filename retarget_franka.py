"""将短视频中的左右手腕轨迹重定向到 Franka 风格机械臂。

脚本会先用 MediaPipe 从输入视频中分别提取左手和右手的腕部轨迹，
再把相机归一化坐标映射到 MuJoCo 中一个安全、有限的 Franka 工作空间。
每只手会对应一台独立的 Franka 风格机械臂，便于并排观察两条轨迹的跟踪效果。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import mujoco
import numpy as np
import matplotlib.pyplot as plt


FRANKA_XML = r"""
<mujoco model="franka_trajectory_retarget">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <asset>
    <material name="white" rgba=".85 .87 .9 1"/><material name="dark" rgba=".12 .15 .18 1"/>
    <material name="blue" rgba=".1 .45 .9 1"/><material name="target" rgba="1 .2 .12 .75"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".18 .2 .22" rgb2=".25 .27 .29"
             width="512" height="512"/><material name="floor" texture="grid" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="1 -1 2.5" dir="-.4 .3 -1" diffuse=".9 .9 .9"/>
    <geom type="plane" size="2 2 .1" material="floor"/>
    <camera name="view" pos="1.65 -1.7 1.25" xyaxes=".72 .69 0 -.28 .3 .91"/>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size=".12 .08" pos="0 0 .08" material="dark"/>
      <body name="link1" pos="0 0 .16">
        <joint name="j1" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="capsule" fromto="0 0 0 0 0 .333" size=".055" material="white"/>
        <body name="link2" pos="0 0 .333">
          <joint name="j2" axis="0 1 0" range="-1.7628 1.7628"/>
          <geom type="capsule" fromto="0 0 0 0 0 .316" size=".05" material="dark"/>
          <body name="link3" pos="0 0 .316">
            <joint name="j3" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="capsule" fromto="0 0 0 .0825 0 .2" size=".047" material="white"/>
            <body name="link4" pos=".0825 0 .2">
              <joint name="j4" axis="0 -1 0" range="-3.0718 -.0698"/>
              <geom type="capsule" fromto="0 0 0 -.0825 0 .184" size=".045" material="dark"/>
              <body name="link5" pos="-.0825 0 .184">
                <joint name="j5" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="capsule" fromto="0 0 0 0 0 .22" size=".042" material="white"/>
                <body name="link6" pos="0 0 .22">
                  <joint name="j6" axis="0 1 0" range="-0.0175 3.7525"/>
                  <geom type="capsule" fromto="0 0 0 .088 0 .107" size=".038" material="dark"/>
                  <body name="link7" pos=".088 0 .107">
                    <joint name="j7" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="capsule" fromto="0 0 0 0 0 .103" size=".04" material="white"/>
                    <site name="ee" pos="0 0 .103" size=".028" material="blue"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="target" mocap="true"><geom type="sphere" size=".035" material="target" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>
"""


def smooth_track(values: np.ndarray, valid: np.ndarray, window: int = 7) -> np.ndarray:
    """补齐并平滑一只手的腕部轨迹。

    MediaPipe 在个别帧可能漏检，原始轨迹里会出现 NaN。
    这里先对每个坐标轴做线性插值，把漏检点接回连续轨迹；
    然后使用简单滑动平均去掉检测抖动，让后面的机械臂逆运动学更稳定。
    """
    x = np.arange(len(values))
    out = values.copy()
    good = np.flatnonzero(valid)
    for axis in range(3):
        out[:, axis] = np.interp(x, good, values[good, axis])
        padded = np.pad(out[:, axis], (window // 2, window // 2), mode="edge")
        out[:, axis] = np.convolve(padded, np.ones(window) / window, mode="valid")
    return out


def extract_wrist(video: Path, model: Path, sample_fps: float):
    """从输入视频中抽样提取左右手腕关键点。

    返回内容包括：原视频帧率、被采样的原始帧号、采样帧图像，
    以及 Left/Right 两只手各自的腕部轨迹、检出掩码和置信度。
    这里保留 MediaPipe 的 handedness 标签；如果视频是镜像画面，
    Left/Right 的语义可能需要按实际画面再人工确认。
    """
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    stride = max(1, round(fps / sample_fps))
    frames = np.arange(0, count, stride, dtype=int)
    tracks = {"Left": np.full((len(frames), 3), np.nan), "Right": np.full((len(frames), 3), np.nan)}
    scores = {"Left": np.zeros(len(frames)), "Right": np.zeros(len(frames))}
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=.45, min_hand_presence_confidence=.45,
        min_tracking_confidence=.45,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)
    thumbnails = []
    for i, frame_no in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = cap.read()
        if not ok:
            thumbnails.append(np.zeros((height, width, 3), np.uint8)); continue
        small = cv2.resize(frame, (640, round(height * 640 / width))) if width > 640 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(frame_no * 1000 / fps))
        for landmarks, handed in zip(result.hand_landmarks, result.handedness):
            label, score = handed[0].category_name, float(handed[0].score)
            if score > scores[label][i]:
                tracks[label][i] = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
                scores[label][i] = score
        thumbnails.append(frame)
        if (i + 1) % 50 == 0:
            print(f"detect {i + 1}/{len(frames)}", flush=True)
    detector.close(); cap.release()
    output = {}
    for hand in ("Left", "Right"):
        valid = np.isfinite(tracks[hand][:, 0])
        if valid.sum() < 3:
            raise RuntimeError(f"{hand} 手的检出帧太少，无法形成可靠轨迹。")
        output[hand] = {
            "wrist": smooth_track(tracks[hand], valid),
            "detected": valid,
            "confidence": scores[hand],
        }
    return fps, frames, thumbnails, output


def map_to_workspace(wrist: np.ndarray) -> np.ndarray:
    """把图像中的腕部轨迹映射到 Franka 可到达的三维工作空间。

    映射时不直接使用手在画面里的绝对位置，而是先按轨迹的 5%~95%
    分位范围做归一化；这样可以保留“移动形状”，同时避免画面边缘或
    镜头位置导致机械臂目标点跑出可达范围。
    """
    lo, hi = np.percentile(wrist[:, :2], [5, 95], axis=0)
    center = (lo + hi) / 2
    spread = np.maximum(hi - lo, .08)
    xy = (wrist[:, :2] - center) / spread
    target = np.empty_like(wrist)
    target[:, 0] = .48 - .26 * xy[:, 1]  # 图像向上移动，对应机械臂向前伸
    target[:, 1] = -.30 * xy[:, 0]       # 图像向右移动，对应机械臂向右侧移动
    z_signal = wrist[:, 2] - np.median(wrist[:, 2])
    target[:, 2] = .62 + np.clip(-z_signal * 1.2, -.10, .10)
    return target


def make_arm():
    """创建一套 MuJoCo Franka 风格机械臂、数据区和渲染器。

    这里使用轻量化 XML 模型，重点是末端点轨迹跟踪，不追求完整的
    Panda 机械结构细节。初始关节角选在较自然的前伸姿态，减少 IK
    一开始陷入关节极限的概率。
    """
    model = mujoco.MjModel.from_xml_string(FRANKA_XML)
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, -.35, 0, -2.1, 0, 1.8, .75]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    renderer = mujoco.Renderer(model, 480, 640)
    return model, data, site, renderer


def ik_step(model, data, site, target):
    """执行一帧末端位置逆运动学。

    使用阻尼最小二乘法根据末端位置误差更新 7 个关节角。
    每次更新都会裁剪步长和关节范围，避免轨迹中突然的大跳变。
    """
    for _ in range(35):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[site]
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 2e-4 * np.eye(3), err)
        data.qpos[:7] += np.clip(dq, -.08, .08)
        data.qpos[:7] = np.clip(data.qpos[:7], model.jnt_range[:7, 0], model.jnt_range[:7, 1])
        if np.linalg.norm(err) < 5e-4: break
    mujoco.mj_forward(model, data)
    data.mocap_pos[0] = target
    mujoco.mj_forward(model, data)
    return data.site_xpos[site].copy(), data.qpos[:7].copy()


def solve_and_render(
    targets,
    wrists,
    source_frames,
    out_dir: Path,
    sample_fps: float,
    video_stem: str,
):
    """求解左右手对应的机械臂轨迹，并渲染三栏对比视频。

    左栏显示原视频和两只手的腕部轨迹尾迹；中栏显示左手轨迹驱动的
    Franka；右栏显示右手轨迹驱动的 Franka。这样可以同时检查两只手
    是否被稳定跟踪，以及机械臂末端是否贴近目标轨迹。
    """
    arms = {hand: make_arm() for hand in ("Left", "Right")}
    writer = cv2.VideoWriter(str(out_dir / f"{video_stem}_franka_retarget.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (1920, 480))
    actual = {h: [] for h in arms}; errors = {h: [] for h in arms}; qposes = {h: [] for h in arms}
    colors = {"Left": (30, 220, 255), "Right": (255, 170, 30)}
    for i in range(len(source_frames)):
        source = cv2.resize(source_frames[i], (640, 480))
        robot_views = []
        for hand in ("Left", "Right"):
            model, data, site, renderer = arms[hand]
            pos, qpos = ik_step(model, data, site, targets[hand][i])
            actual[hand].append(pos)
            errors[hand].append(float(np.linalg.norm(pos - targets[hand][i])))
            qposes[hand].append(qpos)
            trail = wrists[hand][max(0, i - 35):i + 1]
            trail_px = np.column_stack([trail[:, 0] * 640, trail[:, 1] * 480]).astype(int)
            for a, b in zip(trail_px, trail_px[1:]):
                cv2.line(source, tuple(a), tuple(b), colors[hand], 3, cv2.LINE_AA)
            cv2.circle(source, tuple(trail_px[-1]), 9, colors[hand], -1, cv2.LINE_AA)
            renderer.update_scene(data, camera="view")
            robot = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(robot, f"{hand} hand -> Franka | error {errors[hand][-1]*1000:.2f} mm",
                        (18, 34), 0, .61, colors[hand], 2)
            robot_views.append(robot)
        cv2.putText(source, f"{video_stem}: both wrist trajectories", (18, 34), 0, .72, (255,255,255), 2)
        writer.write(np.hstack([source, *robot_views]))
    writer.release()
    for _, _, _, renderer in arms.values(): renderer.close()
    return ({h: np.asarray(v) for h, v in actual.items()},
            {h: np.asarray(v) for h, v in errors.items()},
            {h: np.asarray(v) for h, v in qposes.items()})


def save_comparison_plot(path: Path, targets: dict, actual: dict, sample_fps: float, video_stem: str):
    """保存目标轨迹与 Franka 末端实际轨迹的 X/Y/Z 对比图。"""
    t = np.arange(len(targets["Left"])) / sample_fps
    fig, axes = plt.subplots(3, 2, figsize=(13, 7), sharex=True)
    for col, hand in enumerate(("Left", "Right")):
        for axis, name in enumerate(("X", "Y", "Z")):
            axes[axis, col].plot(t, targets[hand][:, axis], label="mapped target", linewidth=2)
            axes[axis, col].plot(t, actual[hand][:, axis], "--", label="Franka EE", linewidth=1.4)
            axes[axis, col].set_ylabel(f"{name} (m)"); axes[axis, col].grid(alpha=.25)
        axes[0, col].set_title(f"{hand} hand"); axes[0, col].legend()
        axes[-1, col].set_xlabel("time (s)")
    fig.suptitle(f"{video_stem} both hand trajectories vs. Franka end-effectors")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    """命令行入口：提取轨迹、映射工作空间、求解 IK，并保存所有结果。"""
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="dataset/short1.mp4")
    p.add_argument("--model", default="models/hand_landmarker.task")
    p.add_argument("--output", default="output/short1/franka_retarget")
    p.add_argument("--sample-fps", type=float, default=15)
    args = p.parse_args()
    video_path = Path(args.video)
    video_stem = video_path.stem
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fps, frames, images, hand_data = extract_wrist(
        video_path, Path(args.model), args.sample_fps)
    wrists = {h: hand_data[h]["wrist"] for h in ("Left", "Right")}
    targets = {h: map_to_workspace(wrists[h]) for h in ("Left", "Right")}
    actual, errors, qposes = solve_and_render(targets, wrists, images, out, args.sample_fps, video_stem)
    save_comparison_plot(out / "trajectory_comparison.png", targets, actual, args.sample_fps, video_stem)
    with (out / "trajectory_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["frame", "time_sec", "hand", "detected", "confidence",
                  "hand_x", "hand_y", "hand_z", "target_x", "target_y", "target_z",
                  "actual_x", "actual_y", "actual_z", "error_m"] + [f"q{i}" for i in range(1, 8)]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for hand in ("Left", "Right"):
            for i in range(len(frames)):
                detected, confidence = hand_data[hand]["detected"], hand_data[hand]["confidence"]
                wrist = wrists[hand]
                row = dict(frame=int(frames[i]), time_sec=frames[i]/fps, hand=hand,
                           detected=bool(detected[i]), confidence=confidence[i],
                           hand_x=wrist[i,0], hand_y=wrist[i,1], hand_z=wrist[i,2],
                           target_x=targets[hand][i,0], target_y=targets[hand][i,1], target_z=targets[hand][i,2],
                           actual_x=actual[hand][i,0], actual_y=actual[hand][i,1], actual_z=actual[hand][i,2],
                           error_m=errors[hand][i])
                row.update({f"q{j+1}": qposes[hand][i,j] for j in range(7)}); w.writerow(row)
    summary = {
        "video": str(Path(args.video).resolve()), "hands": {},
        "sample_fps": args.sample_fps, "samples_per_hand": len(frames),
        "mapping": "relative wrist motion -> bounded Franka Cartesian workspace; damped least-squares IK",
    }
    for hand in ("Left", "Right"):
        e = errors[hand]
        summary["hands"][hand] = {
            "detection_ratio": round(float(hand_data[hand]["detected"].mean()), 4),
            "mean_tracking_error_mm": round(float(e.mean()*1000), 6),
            "max_tracking_error_mm": round(float(e.max()*1000), 6),
            "rmse_tracking_error_mm": round(float(np.sqrt(np.mean(e**2))*1000), 6),
        }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(out / "trajectory_data.npz",
             left_hand=wrists["Left"], right_hand=wrists["Right"],
             left_target=targets["Left"], right_target=targets["Right"],
             left_actual=actual["Left"], right_actual=actual["Right"],
             left_qpos=qposes["Left"], right_qpos=qposes["Right"],
             left_errors=errors["Left"], right_errors=errors["Right"])
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
