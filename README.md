# 第一视角视频切分与手部轨迹重定向

本项目包含两条主要流程：

- 对 `long1/long2/long3` 这类长视频，先按环境视觉变化做粗切分，再用 MediaPipe 检测手部，只保留稳定出现手的有效区间。
- 对 `short1/short2` 这类短视频，提取左右手腕轨迹，并在 MuJoCo 中映射到两台 Franka 风格机械臂上做双手 retarget 可视化。

## 安装

建议使用 Python 3.10 或 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 运行长视频切分

```powershell
python segment_videos.py dataset/long1.mp4 --clean
python segment_videos.py dataset/long2.mp4 --clean
python segment_videos.py dataset/long3.mp4 --clean
```

若 Windows 上的 MediaPipe 退出阶段被遥测线程阻塞，可安全地拆成两步：

```powershell
python segment_videos.py dataset/long2.mp4 --clean --detect-only
python segment_videos.py dataset/long2.mp4 --reuse-hand-mask
```

仓库已包含 MediaPipe 所需的 `models/hand_landmarker.task`。若模型放在其他位置，可传入
`--hand-model 模型路径`。

默认会按输入视频名输出到 `output/视频名/`，例如 `output/long2/`：

- `clips/segment_*.mp4`：有效片段；
- `long1_hand_segments.mp4`：按时间顺序拼接的有效片段；
- `segments.json`：场景边界、参数、片段时间戳和手部检出率；
- `timeline.png`：蓝色为环境边界，绿色为保留片段；

- `hand_presence.npy`：逐帧手部存在标记缓存，方便调参时用 `--reuse-hand-mask` 跳过耗时的 MediaPipe 检测，也便于后续诊断每一帧是否被判定为有手。

## 参数调节

```powershell
python segment_videos.py dataset/long2.mp4 `
  --scene-threshold 0.35 `
  --min-scene-sec 2.0 `
  --max-hand-gap-sec 0.35 `
  --min-hand-sec 0.7 `
  --min-hand-ratio 0.35 `
  --hand-sample-fps 10 `
  --hand-detection-width 640
```

- 不传 `--scene-threshold` 时，脚本会根据当前输入视频的帧间差异自适应计算阈值。
- 环境被切得太碎时提高 `--scene-threshold` 或 `--min-scene-sec`。
- 手部片段漏得较多时提高 `--max-hand-gap-sec`，或降低 `--min-detection-confidence`。
- 需要更高精度时可提高 `--hand-sample-fps` 和 `--hand-detection-width`，代价是运行更慢。
- 当前由 OpenCV 编码输出，保留画面但不保留原音轨。

## 生成片段级中文描述

```powershell
python describe_segments.py
```

脚本对每个片段均匀抽取关键帧，输出场景、手部移动幅度、运动方向和主要手势：

- `output/long1/descriptions/segment_descriptions.json`：完整结构化结果；
- `output/long1/descriptions/segment_descriptions.csv`：便于表格检查；
- `output/long1/descriptions/segment_descriptions.md`：便于直接阅读或写入报告。

场景描述采用保守策略，只描述亮度、色彩、背景纹理和第一视角构图；没有物体识别
模型支持时，不会凭空生成具体物体名称。手势分为张开、握拳/抓握、捏合、食指伸出、
双指伸出和半张开操作姿态，移动幅度则由归一化腕部轨迹计算。

## 提取手部几何、掌心朝向与运动轨迹

```powershell
python extract_hand_geometry.py
```

默认以 10 FPS 处理 long1 的有效片段，输出至 `output/long1/hand_geometry/`：

- `frame_geometry.jsonl`：逐采样帧的 21 点图像/世界坐标、关节角和掌心法向；
- `trajectories.csv`：左右手腕轨迹、速度、姿态和掌心朝向；
- `geometry_summary.json`：路径长度、速度、姿态与朝向分布；
- `wrist_trajectories.png`：二维腕部轨迹图；
- `hand_geometry.mp4`：骨架、掌心法向红箭头和运动速度黄箭头可视化。

如只需要数值结果，可使用 `python extract_hand_geometry.py --no-video`。

## 对短视频做 Franka retarget

```powershell
python retarget_franka.py --video dataset/short1.mp4 --output output/short1/franka_retarget
python retarget_franka.py --video dataset/short2.mp4 --output output/short2/franka_retarget
```

输出包括：

- `视频名_franka_retarget.mp4`：左侧为原视频双手轨迹，中间/右侧分别为左手和右手驱动的 Franka；
- `trajectory_comparison.png`：目标轨迹与机械臂末端轨迹的 X/Y/Z 对比；
- `trajectory_comparison.csv`：逐采样点的左右手轨迹、目标点、末端点、误差和 7 个关节角；
- `trajectory_data.npz`：供后续程序读取的数组数据；
- `metrics.json`：左右手检出率和跟踪误差统计。
