import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


VIDEO_SIZE = (960, 540)
VIDEO_FPS = 10
ANALYSIS_FPS = 5


@dataclass
class PersonStyle:
    name: str
    color: Tuple[int, int, int]


PERSON_STYLES = {
    "张三": PersonStyle("张三", (80, 200, 80)),
    "李四": PersonStyle("李四", (80, 160, 255)),
    "王五": PersonStyle("王五", (255, 180, 80)),
    "赵六": PersonStyle("赵六", (180, 120, 240)),
}


def lerp_point(a: Tuple[float, float], b: Tuple[float, float], ratio: float) -> Tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def time_str(seconds: float) -> str:
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:04.1f}"


def location_config() -> Dict[str, Dict[str, int]]:
    return {
        "P1": {"x": 120, "y": 120},
        "P2": {"x": 320, "y": 120},
        "P3": {"x": 520, "y": 120},
        "P4": {"x": 120, "y": 300},
        "P5": {"x": 320, "y": 300},
        "P6": {"x": 520, "y": 300},
        "P7": {"x": 120, "y": 480},
        "P8": {"x": 320, "y": 480},
        "P9": {"x": 520, "y": 480},
    }


def build_annotations() -> List[List[str]]:
    return [
        ["剧本1_单人静态身份.mp4", "00:00.0", "00:03.0", "初始站位", "张三", "已注册", "P1", "无"],
        ["剧本1_单人静态身份.mp4", "00:04.0", "00:07.0", "中间打卡", "张三", "已注册", "P2", "无"],
        ["剧本1_单人静态身份.mp4", "00:08.0", "00:11.0", "中间打卡", "张三", "已注册", "P3", "无"],
        ["剧本2_单人动态行为.mp4", "00:00.0", "00:03.0", "初始站位", "李四", "已注册", "P1", "无"],
        ["剧本2_单人动态行为.mp4", "00:10.0", "00:15.0", "异常行为", "李四", "已注册", "P5", "摔倒"],
        ["剧本2_单人动态行为.mp4", "00:18.0", "00:23.0", "异常行为", "李四", "已注册", "P8", "抽烟"],
        ["剧本3_多人交叉轨迹.mp4", "00:00.0", "00:03.0", "初始站位", "张三", "已注册", "P1", "无"],
        ["剧本3_多人交叉轨迹.mp4", "00:00.0", "00:03.0", "初始站位", "李四", "已注册", "P3", "无"],
        ["剧本3_多人交叉轨迹.mp4", "00:00.0", "00:03.0", "初始站位", "赵六", "未注册", "P7", "无"],
        ["剧本3_多人交叉轨迹.mp4", "00:20.0", "00:21.0", "中间打卡", "张三", "已注册", "P5", "无"],
        ["剧本3_多人交叉轨迹.mp4", "00:20.0", "00:21.0", "中间打卡", "赵六", "未注册", "P6", "无"],
        ["剧本4_干扰动作误报.mp4", "00:00.0", "00:03.0", "初始站位", "王五", "已注册", "P2", "无"],
        ["剧本4_干扰动作误报.mp4", "00:08.0", "00:11.0", "干扰动作", "王五", "已注册", "P4", "系鞋带"],
        ["剧本4_干扰动作误报.mp4", "00:12.0", "00:14.0", "干扰动作", "王五", "已注册", "P5", "喝水"],
    ]


def make_bbox(x: float, y: float) -> List[int]:
    return [int(x - 20), int(y - 50), int(x + 20), int(y + 50)]


def timestamp_grid(duration: float) -> List[float]:
    step = 1.0 / ANALYSIS_FPS
    count = int(math.floor(duration * ANALYSIS_FPS)) + 1
    return [round(i * step, 3) for i in range(count)]


def person_position_script1(t: float, loc: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[float, float]]:
    p1 = (loc["P1"]["x"], loc["P1"]["y"])
    p2 = (loc["P2"]["x"], loc["P2"]["y"])
    p3 = (loc["P3"]["x"], loc["P3"]["y"])
    if t <= 3:
        pos = p1
    elif t <= 4:
        pos = lerp_point(p1, p2, t - 3)
    elif t <= 7:
        pos = p2
    elif t <= 8:
        pos = lerp_point(p2, p3, t - 7)
    else:
        pos = p3
    return {"张三": pos}


def person_position_script2(t: float, loc: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[float, float]]:
    p1 = (loc["P1"]["x"], loc["P1"]["y"])
    p5 = (loc["P5"]["x"], loc["P5"]["y"])
    p8 = (loc["P8"]["x"], loc["P8"]["y"])
    p9 = (loc["P9"]["x"], loc["P9"]["y"])
    if t <= 3:
        pos = p1
    elif t <= 10:
        pos = lerp_point(p1, p5, (t - 3) / 7)
    elif t <= 15:
        pos = p5
    elif t <= 18:
        pos = lerp_point(p5, p8, (t - 15) / 3)
    elif t <= 23:
        pos = p8
    else:
        pos = lerp_point(p8, p9, min((t - 23) / 2, 1.0))
    return {"李四": pos}


def person_position_script3(t: float, loc: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[float, float]]:
    p1 = (loc["P1"]["x"], loc["P1"]["y"])
    p3 = (loc["P3"]["x"], loc["P3"]["y"])
    p7 = (loc["P7"]["x"], loc["P7"]["y"])
    p4 = (loc["P4"]["x"], loc["P4"]["y"])
    p5 = (loc["P5"]["x"], loc["P5"]["y"])
    p6 = (loc["P6"]["x"], loc["P6"]["y"])
    p9 = (loc["P9"]["x"], loc["P9"]["y"])
    if t <= 3:
        return {"张三": p1, "李四": p3, "赵六": p7}
    if t <= 12:
        r = (t - 3) / 9
        return {
            "张三": lerp_point(p1, p5, r),
            "李四": lerp_point(p3, p4, r),
            "赵六": lerp_point(p7, p6, r),
        }
    if t <= 16:
        r = (t - 12) / 4
        return {
            "张三": lerp_point(p5, p6, r),
            "李四": lerp_point(p4, p5, r),
            "赵六": lerp_point(p6, p5, r),
        }
    if t <= 21:
        return {"张三": p5, "李四": p4, "赵六": p6}
    r = min((t - 21) / 3, 1.0)
    return {
        "张三": lerp_point(p5, p9, r),
        "李四": lerp_point(p4, p1, r),
        "赵六": lerp_point(p6, p3, r),
    }


def person_position_script4(t: float, loc: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[float, float]]:
    p2 = (loc["P2"]["x"], loc["P2"]["y"])
    p4 = (loc["P4"]["x"], loc["P4"]["y"])
    p5 = (loc["P5"]["x"], loc["P5"]["y"])
    p6 = (loc["P6"]["x"], loc["P6"]["y"])
    if t <= 3:
        pos = p2
    elif t <= 8:
        pos = lerp_point(p2, p4, (t - 3) / 5)
    elif t <= 11:
        pos = p4
    elif t <= 12:
        pos = lerp_point(p4, p5, t - 11)
    elif t <= 14:
        pos = p5
    else:
        pos = lerp_point(p5, p6, min((t - 14) / 2, 1.0))
    return {"王五": pos}


def generate_truth_tracks(loc: Dict[str, Dict[str, int]]) -> Dict[str, Dict[float, Dict[str, Tuple[float, float]]]]:
    durations = {
        "剧本1_单人静态身份.mp4": 12.0,
        "剧本2_单人动态行为.mp4": 25.0,
        "剧本3_多人交叉轨迹.mp4": 24.0,
        "剧本4_干扰动作误报.mp4": 16.0,
    }
    mapping = {
        "剧本1_单人静态身份.mp4": person_position_script1,
        "剧本2_单人动态行为.mp4": person_position_script2,
        "剧本3_多人交叉轨迹.mp4": person_position_script3,
        "剧本4_干扰动作误报.mp4": person_position_script4,
    }
    tracks = {}
    for video_name, duration in durations.items():
        tracks[video_name] = {
            ts: mapping[video_name](ts, loc) for ts in timestamp_grid(duration)
        }
    return tracks


def generate_mock_outputs(
    truth_tracks: Dict[str, Dict[float, Dict[str, Tuple[float, float]]]]
) -> Dict[str, List[Dict[str, object]]]:
    outputs: Dict[str, List[Dict[str, object]]] = {}

    for video_name, timeline in truth_tracks.items():
        frames: List[Dict[str, object]] = []
        for ts, positions in timeline.items():
            targets = []
            if video_name == "剧本1_单人静态身份.mp4":
                x, y = positions["张三"]
                name = "张三"
                if 4.2 <= ts <= 5.4:
                    name = "李四"
                targets.append(
                    {
                        "target_id": 101,
                        "x": round(x + 6),
                        "y": round(y - 5),
                        "bbox": make_bbox(x + 6, y - 5),
                        "action": "正常",
                        "recognized_name": name,
                    }
                )
            elif video_name == "剧本2_单人动态行为.mp4":
                x, y = positions["李四"]
                action = "正常"
                if 11.0 <= ts <= 15.0:
                    action = "摔倒"
                targets.append(
                    {
                        "target_id": 201,
                        "x": round(x + 8),
                        "y": round(y + 4),
                        "bbox": make_bbox(x + 8, y + 4),
                        "action": action,
                        "recognized_name": "李四",
                    }
                )
            elif video_name == "剧本3_多人交叉轨迹.mp4":
                zhang = positions["张三"]
                li = positions["李四"]
                zhao = positions["赵六"]

                zhang_target_id = 301
                zhao_target_id = 303
                zhang_name = "张三"
                zhao_name = "未知"
                zhang_pos = zhang
                zhao_pos = zhao

                if 13.0 <= ts <= 15.0:
                    zhang_target_id = 303
                    zhao_target_id = 301
                if 16.0 <= ts <= 18.0:
                    zhang_target_id = 304
                if 20.0 <= ts <= 21.0:
                    zhang_target_id = 302
                    zhang_name = "李四"
                    zhao_name = "张三"
                if 14.0 <= ts <= 14.8:
                    center_x = (zhang[0] + zhao[0]) / 2
                    center_y = (zhang[1] + zhao[1]) / 2
                    targets.append(
                        {
                            "target_id": 303,
                            "x": round(center_x),
                            "y": round(center_y),
                            "bbox": [int(center_x - 35), int(center_y - 55), int(center_x + 35), int(center_y + 55)],
                            "action": "正常",
                            "recognized_name": "未知",
                        }
                    )
                else:
                    targets.append(
                        {
                            "target_id": zhao_target_id,
                            "x": round(zhao_pos[0] - 7),
                            "y": round(zhao_pos[1] + 5),
                            "bbox": make_bbox(zhao_pos[0] - 7, zhao_pos[1] + 5),
                            "action": "正常",
                            "recognized_name": zhao_name,
                        }
                    )

                zhang_x = zhang_pos[0] + 5
                if 17.0 <= ts <= 17.2:
                    zhang_x += 420
                targets.append(
                    {
                        "target_id": zhang_target_id,
                        "x": round(zhang_x),
                        "y": round(zhang_pos[1] - 6),
                        "bbox": make_bbox(zhang_x, zhang_pos[1] - 6),
                        "action": "正常",
                        "recognized_name": zhang_name,
                    }
                )
                targets.append(
                    {
                        "target_id": 302,
                        "x": round(li[0] + 4),
                        "y": round(li[1] + 3),
                        "bbox": make_bbox(li[0] + 4, li[1] + 3),
                        "action": "正常",
                        "recognized_name": "李四",
                    }
                )
            elif video_name == "剧本4_干扰动作误报.mp4":
                x, y = positions["王五"]
                action = "正常"
                if 9.0 <= ts <= 10.0:
                    action = "摔倒"
                targets.append(
                    {
                        "target_id": 401,
                        "x": round(x + 3),
                        "y": round(y + 2),
                        "bbox": make_bbox(x + 3, y + 2),
                        "action": action,
                        "recognized_name": "王五",
                    }
                )
            frames.append({"timestamp": ts, "targets": targets})
        outputs[video_name] = frames
    return outputs


def render_video(
    video_path: str,
    duration: float,
    truth_timeline: Dict[float, Dict[str, Tuple[float, float]]],
    loc: Dict[str, Dict[str, int]],
) -> None:
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        VIDEO_SIZE,
    )
    width, height = VIDEO_SIZE
    points_px = {
        name: (int(data["x"] * 1.2 + 110), int(data["y"] * 0.8 + 60))
        for name, data in loc.items()
    }

    for frame_idx in range(int(duration * VIDEO_FPS)):
        t = round(frame_idx / VIDEO_FPS, 3)
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.rectangle(frame, (70, 35), (760, 500), (220, 220, 220), -1)
        cv2.putText(frame, os.path.basename(video_path), (30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
        cv2.putText(frame, f"time={t:04.1f}s", (780, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        for point_name, point in points_px.items():
            cv2.circle(frame, point, 7, (70, 70, 70), -1)
            cv2.putText(frame, point_name, (point[0] - 18, point[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

        sample_ts = round(round(t * ANALYSIS_FPS) / ANALYSIS_FPS, 3)
        sample_ts = min(sample_ts, max(truth_timeline.keys()))
        people = truth_timeline[sample_ts]
        for name, (x, y) in people.items():
            px = int(x * 1.2 + 110)
            py = int(y * 0.8 + 60)
            style = PERSON_STYLES[name]
            cv2.circle(frame, (px, py), 18, style.color, -1)
            cv2.putText(frame, name, (px - 24, py - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, style.color, 2)

        cv2.putText(frame, "Ground truth actors", (785, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
        legend_y = 120
        for person_name in people:
            style = PERSON_STYLES[person_name]
            cv2.circle(frame, (800, legend_y), 8, style.color, -1)
            cv2.putText(frame, person_name, (820, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 2)
            legend_y += 28

        writer.write(frame)

    writer.release()


def create_workspace(output_dir: str, overwrite: bool) -> None:
    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    videos_dir = os.path.join(output_dir, "videos")
    config_dir = os.path.join(output_dir, "config")
    mock_dir = os.path.join(output_dir, "mock_api_outputs")
    ensure_dir(videos_dir)
    ensure_dir(config_dir)
    ensure_dir(mock_dir)

    loc = location_config()
    with open(os.path.join(config_dir, "location_config.json"), "w", encoding="utf-8") as f:
        json.dump(loc, f, ensure_ascii=False, indent=2)

    annotations = build_annotations()
    annotation_headers = ["视频名称", "开始时间", "结束时间", "事件类型", "真实姓名", "注册状态", "发生位置", "动作标签"]
    with open(os.path.join(output_dir, "annotations.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(annotation_headers)
        writer.writerows(annotations)

    truth_tracks = generate_truth_tracks(loc)
    mock_outputs = generate_mock_outputs(truth_tracks)

    manifest = {
        "analysis_fps": ANALYSIS_FPS,
        "videos": {},
    }

    for video_name, timeline in truth_tracks.items():
        duration = max(timeline.keys()) + 1.0 / ANALYSIS_FPS
        render_video(os.path.join(videos_dir, video_name), duration, timeline, loc)
        with open(os.path.join(mock_dir, f"{os.path.splitext(video_name)[0]}.json"), "w", encoding="utf-8") as f:
            json.dump(mock_outputs[video_name], f, ensure_ascii=False, indent=2)
        manifest["videos"][video_name] = {
            "duration_seconds": duration,
            "mock_output_file": f"{os.path.splitext(video_name)[0]}.json",
        }

    with open(os.path.join(mock_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成视频分析评测模拟环境与样例数据。")
    parser.add_argument("--output_dir", default="eval_workspace", help="输出工作区目录")
    parser.add_argument("--overwrite", action="store_true", help="若目录已存在则覆盖重建")
    args = parser.parse_args()

    create_workspace(args.output_dir, args.overwrite)
    print("环境准备完成。")
    print(f"工作区: {os.path.abspath(args.output_dir)}")
    print("下一步可以执行：")
    print(
        "python run_eval.py "
        f"--video_dir {os.path.join(args.output_dir, 'videos')} "
        f"--anno_file {os.path.join(args.output_dir, 'annotations.csv')} "
        f"--config {os.path.join(args.output_dir, 'config', 'location_config.json')} "
        f"--mock_dir {os.path.join(args.output_dir, 'mock_api_outputs')}"
    )


if __name__ == "__main__":
    main()
