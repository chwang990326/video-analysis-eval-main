import argparse
import base64
import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from openpyxl import Workbook


POSITION_EVENT_TYPES = {"初始站位", "中间打卡"}
ABNORMAL_EVENT_TYPE = "异常行为"
INTERFERENCE_EVENT_TYPE = "干扰动作"
UNKNOWN_NAMES = {"未知", "未注册", "unknown", "Unknown", "stranger", "陌生人", ""}
BEHAVIOR_LABEL_MAP = {
    "climbing": "攀爬",
    "falling_down": "摔倒",
    "looking_at_phone": "看手机",
    "normal": "正常",
    "reaching_high": "摸高",
    "sleeping": "睡觉",
    "smoking": "抽烟",
}

# 自动生成反向字典 (中文 -> 英文)，专用于 OpenCV 渲染
EN_ACTION_MAP = {v: k for k, v in BEHAVIOR_LABEL_MAP.items()}
EN_ACTION_MAP["无"] = "none"
EN_ACTION_MAP["未知"] = "unknown"
EN_ACTION_MAP["正常"] = "normal"

EN_ISSUE_TYPE_MAP = {
    "轨迹跳变": "Track Jump",
    "ID串号": "ID Switch",
    "误合并": "False Merge",
    "ID断裂": "ID Break",
    "行为漏报": "False Negative",
    "行为误报": "False Positive",
    "身份识别错误": "Identity Error",
    "ID变化": "ID Change",
    "检测框断裂": "Detect Lose"
}

def parse_abs_time(time_str: str) -> float:
    """
    解析绝对时间字符串为 Unix 时间戳 (秒)
    支持: 20260409151543617 (17位: 含毫秒) 或 20260409151543 (14位: 精确到秒)
    """
    if time_str is None or str(time_str).strip() == "":
        return 0.0
    time_str = str(time_str).strip()
    if len(time_str) >= 14:
        dt_part = time_str[:14]
        ms_part = time_str[14:] if len(time_str) > 14 else "0"
        ms_val = float(ms_part) / (10 ** len(ms_part))
        # 使用 UTC 时区解析，避免本地机器时区或夏令时的干扰
        dt = datetime.strptime(dt_part, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp() + ms_val
    raise ValueError(f"不支持的绝对时间格式: {time_str}")


def extract_video_start_time(video_name: str) -> float:
    """
    从视频文件名中提取绝对开始时间戳
    例如: NVR_ch1_main_20260408144300_20260408145400.mp4 -> 解析 20260408144300
    """
    match = re.search(r"_(\d{14})_(\d{14})\.[a-zA-Z0-9]+$", video_name)
    if not match:
        raise ValueError(f"视频文件名未包含标准的起止时间戳，无法解析零点: {video_name}")
    start_str = match.group(1)
    dt = datetime.strptime(start_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def calc_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)


@dataclass
class IssueRecord:
    video_name: str
    timestamp: float
    issue_type: str
    detail: str


class DetectionProvider:
    def __init__(self) -> None:
        self.last_raw_response: object = None

    def get_detections(self, video_name: str, timestamp: float, frame: np.ndarray, video_start_ts: float = 0.0) -> List[Dict[str, object]]:
        raise NotImplementedError


class MockDetectionProvider(DetectionProvider):
    def __init__(self, mock_dir: str) -> None:
        super().__init__()
        self.mock_dir = mock_dir
        self.cache: Dict[str, Dict[float, List[Dict[str, object]]]] = {}

    def _load_video(self, video_name: str) -> Dict[float, List[Dict[str, object]]]:
        if video_name not in self.cache:
            base_name = os.path.splitext(video_name)[0]
            path = os.path.join(self.mock_dir, f"{base_name}.json")
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            timeline = {
                round(float(record["timestamp"]), 3): record.get("targets", [])
                for record in records
            }
            self.cache[video_name] = timeline
        return self.cache[video_name]

    def get_detections(self, video_name: str, timestamp: float, frame: np.ndarray, video_start_ts: float = 0.0) -> List[Dict[str, object]]:
        timeline = self._load_video(video_name)
        rounded = round(timestamp, 3)
        self.last_raw_response = {"timestamp": rounded, "targets": timeline.get(rounded)}
        if rounded in timeline:
            return timeline[rounded]
        nearest = min(timeline.keys(), key=lambda key: abs(key - rounded))
        self.last_raw_response = {"timestamp": nearest, "targets": timeline[nearest]}
        return timeline[nearest]


class ApiDetectionProvider(DetectionProvider):
    def __init__(
        self,
        api_url: str,
        timeout: float,
        camera_id: str,
        enable_face_recognition: bool,
        enable_behavior_detection: bool,
        enable_spatial_positioning: bool,
        enable_target_tracking: bool,
        world_coord_scale: float,
    ) -> None:
        super().__init__()
        self.api_url = api_url
        self.timeout = timeout
        self.camera_id = camera_id
        self.enable_face_recognition = enable_face_recognition
        self.enable_behavior_detection = enable_behavior_detection
        self.enable_spatial_positioning = enable_spatial_positioning
        self.enable_target_tracking = enable_target_tracking
        self.world_coord_scale = world_coord_scale
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get_detections(self, video_name: str, timestamp: float, frame: np.ndarray, video_start_ts: float = 0.0) -> List[Dict[str, object]]:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError(f"无法编码视频帧: {video_name}@{timestamp:.3f}s")

        image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        
        # 计算真实的绝对时间戳，并格式化为 ISO 8601 发给 API
        abs_timestamp = video_start_ts + timestamp
        iso_timestamp = datetime.fromtimestamp(abs_timestamp, tz=timezone.utc).isoformat(timespec="milliseconds")
        
        payload = json.dumps(
            {
                "video_name": video_name,
                "timestamp": iso_timestamp,
                "camera_id": self.camera_id,
                "associated_camera_ids": [],
                "image": image_base64,
                "image_base64": image_base64,
                "enable_face_recognition": self.enable_face_recognition,
                "enable_behavior_detection": self.enable_behavior_detection,
                "enable_spatial_positioning": self.enable_spatial_positioning,
                "enable_target_tracking": self.enable_target_tracking,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"调用分析 API 失败: {exc}") from exc
        self.last_raw_response = body

        if not isinstance(body, dict):
            return body if isinstance(body, list) else []
        if body.get("code") not in (None, 0):
            raise RuntimeError(f"分析 API 返回错误: code={body.get('code')}, message={body.get('message')}")
        data = body.get("data", {})
        if not data.get("exist_person", False):
            return []
        persons = data.get("persons", [])
        detections = []
        for index, person in enumerate(persons):
            person_id = person.get("person_id")
            recognized_name = self.normalize_person_name(person_id)
            world_coordinates = person.get("world_coordinates") or [0.0, 0.0, 0.0]
            x, y = self.normalize_world_coordinates(world_coordinates)
            bbox = self.normalize_bbox(person.get("bounding_box"))
            detections.append(
                {
                    "target_id": str(person_id) if person_id is not None else f"unknown_{index}",
                    "x": x,
                    "y": y,
                    "bbox": bbox,
                    "action": self.extract_action(person.get("behavior_events") or []),
                    "recognized_name": recognized_name,
                    "confidence": float(person.get("conf") or 0.0),
                    "keypoint_count": int(person.get("keypoint_count") or 0),
                }
            )
        return detections

    def extract_action(self, behavior_events: List[Dict[str, object]]) -> str:
        if not behavior_events:
            return "正常"
        best = max(behavior_events, key=lambda item: float(item.get("confidence") or 0.0))
        return BEHAVIOR_LABEL_MAP.get(str(best.get("behavior_type", "normal")), str(best.get("behavior_type", "normal")))

    def normalize_person_name(self, person_id: Optional[object]) -> str:
        if person_id is None:
            return "未知"
        text = str(person_id)
        if text.isdigit():
            return "未知"
        return text

    def normalize_world_coordinates(self, world_coordinates: List[object]) -> Tuple[float, float]:
        if len(world_coordinates) < 2:
            return 0.0, 0.0
        return float(world_coordinates[0]) * self.world_coord_scale, float(world_coordinates[1]) * self.world_coord_scale

    def normalize_bbox(self, bounding_box: Optional[List[object]]) -> List[int]:
        if not bounding_box or len(bounding_box) < 4:
            return [0, 0, 0, 0]
        x, y, width, height = [float(v) for v in bounding_box[:4]]
        return [int(x), int(y), int(x + width), int(y + height)]


class VideoEvaluator:
    def __init__(
        self,
        video_dir: str,
        anno_file: str,
        config_file: str,
        output_dir: str,
        provider: DetectionProvider,
        analysis_fps: float,
        jump_speed_threshold: float,
        match_distance_threshold: float,
        merge_distance_threshold: float,
        tracking_mode: str = "single",
    ) -> None:
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.bad_case_dir = os.path.join(output_dir, "bad_cases")
        self.api_dump_dir = os.path.join(output_dir, "api_responses")
        
        self.loc_error_dir = os.path.join(self.bad_case_dir, "localization_errors")
        self.track_jump_dir = os.path.join(self.bad_case_dir, "Track Jump")
        self.id_switch_dir = os.path.join(self.bad_case_dir, "ID Switch")
        self.false_merge_dir = os.path.join(self.bad_case_dir, "False Merge")
        self.id_break_dir = os.path.join(self.bad_case_dir, "ID Break")
        self.id_change_dir = os.path.join(self.bad_case_dir, "id_change") # 新增 ID变化 文件夹
        self.detect_lose_dir = os.path.join(self.bad_case_dir, "detect_lose") # 新增 检测框断裂 文件夹
        
        # 新增的漏报和误报文件夹
        self.fn_dir = os.path.join(self.bad_case_dir, "False Negative")
        self.fp_dir = os.path.join(self.bad_case_dir, "False Positive")
        
        self.provider = provider
        self.analysis_fps = analysis_fps
        self.jump_speed_threshold = jump_speed_threshold
        self.match_distance_threshold = match_distance_threshold
        self.merge_distance_threshold = merge_distance_threshold
        self.tracking_mode = tracking_mode

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.bad_case_dir, exist_ok=True)
        os.makedirs(self.api_dump_dir, exist_ok=True)
        os.makedirs(self.loc_error_dir, exist_ok=True)
        os.makedirs(self.track_jump_dir, exist_ok=True)
        os.makedirs(self.id_switch_dir, exist_ok=True)
        os.makedirs(self.false_merge_dir, exist_ok=True)
        os.makedirs(self.id_break_dir, exist_ok=True)
        os.makedirs(self.id_change_dir, exist_ok=True)
        os.makedirs(self.detect_lose_dir, exist_ok=True)
        os.makedirs(self.fn_dir, exist_ok=True)
        os.makedirs(self.fp_dir, exist_ok=True)

        self.annotations = self.load_annotations(anno_file)

        with open(config_file, "r", encoding="utf-8") as f:
            self.location_config = json.load(f)

        self.summary_rows: List[Dict[str, object]] = []
        self.detail_rows: List[Dict[str, object]] = []
        self.issues: List[IssueRecord] = []
        
        self.loc_error_cases: List[Dict[str, object]] = []
        self.track_jump_cases: List[Dict[str, object]] = []
        self.id_switch_cases: List[Dict[str, object]] = []
        self.false_merge_cases: List[Dict[str, object]] = []
        self.id_break_cases: List[Dict[str, object]] = []
        self.id_change_cases: List[Dict[str, object]] = [] # 新增 ID变化 记录
        self.detect_lose_cases: List[Dict[str, object]] = [] # 新增 检测框断裂 记录
        self.fn_cases: List[Dict[str, object]] = [] # 漏报全时段记录
        self.fp_cases: List[Dict[str, object]] = [] # 误报单帧记录

        self.localization_errors: List[float] = []
        self.track_jump_count = 0
        self.id_switch_count = 0
        self.false_merge_count = 0
        self.id_break_count = 0
        self.id_change_count = 0 # 新增 ID变化 统计
        self.detect_lose_count = 0 # 新增 检测框断裂 统计

        self.behavior_tp = 0
        self.behavior_fn = 0
        self.behavior_fp = 0
        self.behavior_latency: List[float] = []
        self.behavior_owner_correct = 0

        self.registered_id_correct = 0
        self.registered_id_total = 0
        self.unknown_reject_correct = 0
        self.unknown_reject_total = 0
        self.identity_latency: List[float] = []

    def run(self) -> None:
        for video_name in self.unique_video_names():
            print(f"正在评测: {video_name} (追踪模式: {self.tracking_mode})")
            self.process_video(video_name)
        self.write_reports()

    def process_video(self, video_name: str) -> None:
        video_path = os.path.join(self.video_dir, video_name)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"未找到视频文件: {video_path}")

        # 解析当前视频的绝对时间零点
        video_start_ts = extract_video_start_time(video_name)

        # 构建当前视频专用的标注：将绝对时间映射为视频内的相对秒数
        annos = []
        for row in self.annotations:
            if row["视频名称"] == video_name:
                anno = dict(row)
                anno["开始时间_s"] = max(0.0, anno["开始时间_abs"] - video_start_ts)
                anno["结束时间_s"] = max(0.0, anno["结束时间_abs"] - video_start_ts)
                annos.append(anno)

        sampled = self.collect_detections(video_name, video_path, video_start_ts)
        id_mapping = self.bind_target_ids(annos, sampled)
        self.evaluate_positions(video_name, annos, sampled, id_mapping)
        self.evaluate_tracking(video_name, annos, sampled, id_mapping)
        self.evaluate_behaviors(video_name, annos, sampled, id_mapping)
        self.evaluate_identity(video_name, annos, sampled, id_mapping)
        
        self.generate_bad_case_clips(video_name, video_path, sampled)
        self.generate_localization_error_images(video_name, video_path) 
        self.generate_tracking_error_images(video_name, video_path)
        self.generate_id_change_images(video_name, video_path) # 生成 ID变化 图像
        self.generate_detect_lose_images(video_name, video_path) # 生成 检测框断裂 图像
        self.generate_fn_videos(video_name, video_path, sampled) # 漏报存视频
        self.generate_fp_images(video_name, video_path) # 误报存图片

    def collect_detections(self, video_name: str, video_path: str, video_start_ts: float) -> List[Dict[str, object]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0.0
        step = 1.0 / self.analysis_fps
        sample_timestamps = []
        current = 0.0
        while current <= duration + 1e-6:
            sample_timestamps.append(round(current, 3))
            current += step

        sampled: List[Dict[str, object]] = []
        for ts in sample_timestamps:
            frame_index = min(int(round(ts * fps)), max(frame_count - 0, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            detections = self.provider.get_detections(video_name, ts, frame, video_start_ts)
            raw_response = self.provider.last_raw_response
            normalized = []
            for det in detections:
                norm = dict(det)
                norm["target_id"] = str(norm["target_id"])
                norm["x"] = float(norm["x"])
                norm["y"] = float(norm["y"])
                norm["action"] = norm.get("action", "正常")
                norm["recognized_name"] = str(norm.get("recognized_name", ""))
                if "bbox" not in norm:
                    norm["bbox"] = [int(norm["x"] - 20), int(norm["y"] - 50), int(norm["x"] + 20), int(norm["y"] + 50)]
                normalized.append(norm)
            sampled.append(
                {
                    "timestamp": ts,
                    "frame_index": frame_index,
                    "detections": normalized,
                    "raw_response": raw_response,
                }
            )
        cap.release()
        self.write_api_dump(video_name, sampled)
        return sampled

    def bind_target_ids(
        self,
        annos: List[Dict[str, object]],
        sampled: List[Dict[str, object]],
    ) -> Dict[str, str]:
        candidate_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for anno in annos:
            if anno["事件类型"] != "初始站位":
                continue
            gt = self.location_to_xy(anno["发生位置"])
            for sample in self.samples_in_window(sampled, anno["开始时间_s"], anno["结束时间_s"]):
                for det in sample["detections"]:
                    dist = calc_distance((det["x"], det["y"]), gt)
                    if dist <= self.match_distance_threshold:
                        candidate_scores[anno["真实姓名"]][det["target_id"]].append(dist)

        assignments: Dict[str, str] = {}
        used_targets = set()
        people = sorted(candidate_scores.keys(), key=lambda name: len(candidate_scores[name]), reverse=True)
        for person in people:
            scored = []
            for target_id, distances in candidate_scores[person].items():
                avg_dist = sum(distances) / len(distances)
                scored.append((len(distances), -avg_dist, target_id))
            for _, _, target_id in sorted(scored, reverse=True):
                if target_id not in used_targets:
                    assignments[person] = target_id
                    used_targets.add(target_id)
                    break
        return assignments

    def evaluate_positions(
        self,
        video_name: str,
        annos: List[Dict[str, object]],
        sampled: List[Dict[str, object]],
        id_mapping: Dict[str, str],
    ) -> None:
        for anno in annos:
            if anno["事件类型"] not in POSITION_EVENT_TYPES:
                continue
            person = anno["真实姓名"]
            gt = self.location_to_xy(anno["发生位置"])
            target_id = id_mapping.get(person)
            window_errors = []
            for sample in self.samples_in_window(sampled, anno["开始时间_s"], anno["结束时间_s"]):
                det = self.find_detection(sample["detections"], target_id, gt)
                if det is not None:
                    err = calc_distance((det["x"], det["y"]), gt)
                    window_errors.append(err)
                    self.localization_errors.append(err)
                    
                    if err > 30.0:
                        self.loc_error_cases.append({
                            "video_name": video_name,
                            "timestamp": sample["timestamp"],
                            "frame_index": sample["frame_index"],
                            "person": person,
                            "det": det,
                            "gt": gt,
                            "err": err
                        })
                        
            if window_errors:
                self.detail_rows.append(
                    {
                        "视频名称": video_name,
                        "维度": "定位精度",
                        "对象": person,
                        "事件类型": anno["事件类型"],
                        "位置": anno["发生位置"],
                        "结果": round(float(np.median(window_errors)), 2),
                        "附加信息": "窗口中位误差(cm)",
                    }
                )

    def evaluate_tracking(
        self,
        video_name: str,
        annos: List[Dict[str, object]],
        sampled: List[Dict[str, object]],
        id_mapping: Dict[str, str],
    ) -> None:
        if annos:
            jump_eval_start = annos[0]["开始时间_s"]
            jump_eval_end = annos[-1]["结束时间_s"]
            jump_sampled = self.samples_in_window(sampled, jump_eval_start, jump_eval_end)
        else:
            jump_sampled = sampled

        # =====================================================================
        # [新增] 检测框断裂 检测逻辑 (限制在标注起止时间内)
        # =====================================================================
        expected_person_count = len(set(anno["真实姓名"] for anno in annos)) if annos else 1
        
        for sample in jump_sampled:
            det_count = len(sample["detections"])
            is_lose = False
            reason = ""
            
            if self.tracking_mode == "single":
                if det_count == 0:
                    is_lose = True
                    reason = "单人模式: 检测框个数为0"
            elif self.tracking_mode == "multi":
                if det_count < expected_person_count:
                    is_lose = True
                    reason = f"多人模式: 检测框个数({det_count})小于标注人数({expected_person_count})"
                    
            if is_lose:
                self.detect_lose_count += 1
                self.issues.append(IssueRecord(video_name, sample["timestamp"], "检测框断裂", reason))
                self.detect_lose_cases.append({
                    "video_name": video_name,
                    "ts": sample["timestamp"],
                    "frame_index": sample["frame_index"],
                    "det_count": det_count,
                    "expected_count": expected_person_count if self.tracking_mode == "multi" else 1,
                    "reason": reason,
                    "detections": sample["detections"]
                })
        
        # =====================================================================

        trajectories: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for sample in jump_sampled:
            for det in sample["detections"]:
                trajectories[det["target_id"]].append({
                    "ts": sample["timestamp"],
                    "frame_index": sample["frame_index"],
                    "det": det
                })

        for target_id, points in trajectories.items():
            for prev, curr in zip(points, points[1:]):
                dt = curr["ts"] - prev["ts"]
                if dt <= 0:
                    continue
                speed = calc_distance((prev["det"]["x"], prev["det"]["y"]), (curr["det"]["x"], curr["det"]["y"])) / dt
                if speed > self.jump_speed_threshold:
                    self.track_jump_count += 1
                    self.issues.append(IssueRecord(video_name, curr["ts"], "轨迹跳变", f"Target_ID={target_id}, speed={speed:.1f}cm/s"))
                    self.track_jump_cases.append({
                        "video_name": video_name,
                        "ts": curr["ts"],
                        "prev_frame_idx": prev["frame_index"],
                        "curr_frame_idx": curr["frame_index"],
                        "prev_det": prev["det"],
                        "curr_det": curr["det"],
                    })

        # =====================================================================
        # ID变化 检测逻辑 (支持单人 single / 多人 multi)
        # =====================================================================
        if self.tracking_mode == "single":
            # 单人模式：比较有目标的前后两帧，如果 ID 发生了变化，则记录一次
            for prev_s, curr_s in zip(jump_sampled, jump_sampled[1:]):
                if prev_s["detections"] and curr_s["detections"]:
                    # 假定画面中为单人，取置信度最高的目标进行比较
                    prev_det = max(prev_s["detections"], key=lambda x: x.get("confidence", 0))
                    curr_det = max(curr_s["detections"], key=lambda x: x.get("confidence", 0))
                    if prev_det["target_id"] != curr_det["target_id"]:
                        self.id_change_count += 1
                        self.issues.append(IssueRecord(video_name, curr_s["timestamp"], "ID变化", f"单人模式ID变化: {prev_det['target_id']} -> {curr_det['target_id']}"))
                        self.id_change_cases.append({
                            "mode": "single",
                            "video_name": video_name,
                            "ts": curr_s["timestamp"],
                            "prev_frame_idx": prev_s["frame_index"],
                            "curr_frame_idx": curr_s["frame_index"],
                            "prev_det": prev_det,
                            "curr_det": curr_det
                        })
        elif self.tracking_mode == "multi":
            # 多人模式：仅在“中间打卡”时间段内，寻找离打卡点最近的人，与基准 expected_target_id 进行比较
            for anno in annos:
                if anno["事件类型"] != "中间打卡":
                    continue
                person = anno["真实姓名"]
                expected_target_id = id_mapping.get(person)
                gt = self.location_to_xy(anno["发生位置"])
                change_recorded = False
                for sample in self.samples_in_window(sampled, anno["开始时间_s"], anno["结束时间_s"]):
                    nearest = self.nearest_detection(sample["detections"], gt)
                    if nearest is None:
                        continue
                    if expected_target_id and nearest["target_id"] != expected_target_id:
                        if not change_recorded:
                            self.id_change_count += 1
                            self.issues.append(IssueRecord(video_name, sample["timestamp"], "ID变化", f"多人模式ID变化: 本该是 {expected_target_id}, 测出 {nearest['target_id']}"))
                            self.id_change_cases.append({
                                "mode": "multi",
                                "video_name": video_name,
                                "ts": sample["timestamp"],
                                "frame_index": sample["frame_index"],
                                "det": nearest,
                                "expected_id": expected_target_id
                            })
                            change_recorded = True

        for anno in annos:
            if anno["事件类型"] != "中间打卡":
                continue
            person = anno["真实姓名"]
            expected_target_id = id_mapping.get(person)
            gt = self.location_to_xy(anno["发生位置"])
            mismatched = False
            switch_recorded = False
            for sample in self.samples_in_window(sampled, anno["开始时间_s"], anno["结束时间_s"]):
                nearest = self.nearest_detection(sample["detections"], gt)
                if nearest is None:
                    continue
                if expected_target_id and nearest["target_id"] != expected_target_id:
                    mismatched = True
                    if not switch_recorded:
                        self.id_switch_cases.append({
                            "video_name": video_name,
                            "ts": sample["timestamp"],
                            "frame_index": sample["frame_index"],
                            "det": nearest,
                            "expected_id": expected_target_id
                        })
                        switch_recorded = True
            if mismatched:
                self.id_switch_count += 1
                self.issues.append(IssueRecord(video_name, anno["开始时间_s"], "ID串号", f"{person} 在 {anno['发生位置']} 的目标 ID 不一致"))

        for sample in sampled:
            active_positions = [
                row
                for row in annos
                if row["事件类型"] in POSITION_EVENT_TYPES
                and row["开始时间_s"] <= sample["timestamp"] <= row["结束时间_s"]
            ]
            if len(active_positions) < 2:
                continue
            gt_points = [self.location_to_xy(row["发生位置"]) for row in active_positions]
            if len(sample["detections"]) < len(gt_points):
                for det in sample["detections"]:
                    close_points = [
                        point
                        for point in gt_points
                        if calc_distance((det["x"], det["y"]), point) <= self.merge_distance_threshold
                    ]
                    if len(close_points) >= 2:
                        self.false_merge_count += 1
                        self.issues.append(IssueRecord(video_name, sample["timestamp"], "误合并", "检测结果少于真实人数，且单框覆盖多个点位"))
                        self.false_merge_cases.append({
                            "video_name": video_name,
                            "ts": sample["timestamp"],
                            "frame_index": sample["frame_index"],
                            "det": det
                        })
                        break

        break_flags = set()
        for anno in annos:
            if anno["事件类型"] not in POSITION_EVENT_TYPES:
                continue
            person = anno["真实姓名"]
            expected_target_id = id_mapping.get(person)
            gt = self.location_to_xy(anno["发生位置"])
            if not expected_target_id:
                continue
            in_break = False
            last_seen_sample = None
            last_seen_det = None
            for sample in self.samples_in_window(sampled, anno["开始时间_s"], anno["结束时间_s"]):
                det = self.find_by_target_id(sample["detections"], expected_target_id)
                nearest = self.nearest_detection(sample["detections"], gt)
                broken = det is None and nearest is not None and calc_distance((nearest["x"], nearest["y"]), gt) <= self.match_distance_threshold
                if broken and not in_break:
                    key = (video_name, person, sample["timestamp"])
                    if key not in break_flags:
                        self.id_break_count += 1
                        self.issues.append(IssueRecord(video_name, sample["timestamp"], "ID断裂", f"{person} 原始 ID 消失并被新 ID 替代"))
                        break_flags.add(key)
                        
                        if last_seen_sample is not None:
                            self.id_break_cases.append({
                                "video_name": video_name,
                                "ts": sample["timestamp"],
                                "prev_frame_idx": last_seen_sample["frame_index"],
                                "curr_frame_idx": sample["frame_index"],
                                "prev_det": last_seen_det,
                                "curr_det": nearest
                            })
                    in_break = True
                if not broken:
                    in_break = False
                    
                if det is not None:
                    last_seen_sample = sample
                    last_seen_det = det

    def evaluate_behaviors(
        self,
        video_name: str,
        annos: List[Dict[str, object]],
        sampled: List[Dict[str, object]],
        id_mapping: Dict[str, str],
    ) -> None:
        abnormal_annos = [row for row in annos if row["事件类型"] == ABNORMAL_EVENT_TYPE]
        for anno in abnormal_annos:
            person = anno["真实姓名"]
            expected_target_id = id_mapping.get(person)
            gt = self.location_to_xy(anno["发生位置"])
            first_hit_ts = None
            owner_hit = False
            for sample in self.samples_in_window(sampled, anno["开始时间_s"], anno["结束时间_s"]):
                matching = [
                    det
                    for det in sample["detections"]
                    if det["action"] == anno["动作标签"]
                    and calc_distance((det["x"], det["y"]), gt) <= self.match_distance_threshold * 1.5
                ]
                if matching and first_hit_ts is None:
                    first_hit_ts = sample["timestamp"]
                    owner_hit = any(det["target_id"] == expected_target_id for det in matching if expected_target_id)
            
            # 记录漏报全时间段信息
            if first_hit_ts is None:
                self.behavior_fn += 1
                self.issues.append(IssueRecord(video_name, anno["开始时间_s"], "行为漏报", f"{person} 的 {anno['动作标签']} 未检出"))
                self.fn_cases.append({
                    "video_name": video_name,
                    "start_s": anno["开始时间_s"],
                    "end_s": anno["结束时间_s"],
                    "person": person,
                    "action": anno["动作标签"]
                })
            else:
                self.behavior_tp += 1
                self.behavior_latency.append(first_hit_ts - anno["开始时间_s"])
                if owner_hit:
                    self.behavior_owner_correct += 1
                self.detail_rows.append(
                    {
                        "视频名称": video_name,
                        "维度": "行为识别",
                        "对象": person,
                        "事件类型": anno["动作标签"],
                        "位置": anno["发生位置"],
                        "结果": round(first_hit_ts - anno["开始时间_s"], 2),
                        "附加信息": "报警延迟(s)",
                    }
                )

        # =====================================================================
        # 限定行为误报的检测窗口 (从标注起始到标注结束)
        # =====================================================================
        if annos:
            fp_eval_start = annos[0]["开始时间_s"]
            fp_eval_end = annos[-1]["结束时间_s"]
            fp_sampled = self.samples_in_window(sampled, fp_eval_start, fp_eval_end)
        else:
            fp_sampled = sampled

        for sample in fp_sampled:
            active_abnormal = [
                row
                for row in annos
                if row["事件类型"] == ABNORMAL_EVENT_TYPE
                and row["开始时间_s"] <= sample["timestamp"] <= row["结束时间_s"]
            ]
            active_interference = [
                row
                for row in annos
                if row["事件类型"] == INTERFERENCE_EVENT_TYPE
                and row["开始时间_s"] <= sample["timestamp"] <= row["结束时间_s"]
            ]
            allowed_actions = {row["动作标签"] for row in active_abnormal}
            
            for det in sample["detections"]:
                action = det["action"]
                if action in {"正常", "无", ""}:
                    continue
                
                # 记录误报单帧信息
                if not active_interference and action not in allowed_actions:
                    self.behavior_fp += 1
                    reason = f"检测到额外动作 {action}"
                    self.issues.append(IssueRecord(video_name, sample["timestamp"], "行为误报", reason))
                    self.fp_cases.append({
                        "video_name": video_name,
                        "ts": sample["timestamp"],
                        "frame_index": sample["frame_index"],
                        "det": det,
                        "reason": reason
                    })
                elif active_interference:
                    self.behavior_fp += 1
                    reason = f"干扰动作期间误报为 {action}"
                    self.issues.append(IssueRecord(video_name, sample["timestamp"], "行为误报", reason))
                    self.fp_cases.append({
                        "video_name": video_name,
                        "ts": sample["timestamp"],
                        "frame_index": sample["frame_index"],
                        "det": det,
                        "reason": reason
                    })

    def evaluate_identity(
        self,
        video_name: str,
        annos: List[Dict[str, object]],
        sampled: List[Dict[str, object]],
        id_mapping: Dict[str, str],
    ) -> None:
        for anno in annos:
            if anno["事件类型"] not in POSITION_EVENT_TYPES:
                continue
            person = anno["真实姓名"]
            status = anno["注册状态"]
            gt = self.location_to_xy(anno["发生位置"])
            target_id = id_mapping.get(person)
            correct_times = []
            total_checks = 0
            correct_checks = 0
            for sample in self.samples_in_window(sampled, anno["开始时间_s"], anno["结束时间_s"]):
                det = self.find_detection(sample["detections"], target_id, gt)
                if det is None:
                    continue
                total_checks += 1
                recognized_name = det["recognized_name"]
                if status == "已注册":
                    self.registered_id_total += 1
                    if recognized_name == person:
                        self.registered_id_correct += 1
                        correct_checks += 1
                        correct_times.append(sample["timestamp"])
                else:
                    self.unknown_reject_total += 1
                    if recognized_name in UNKNOWN_NAMES:
                        self.unknown_reject_correct += 1
                        correct_checks += 1
            if status == "已注册" and correct_times:
                self.identity_latency.append(correct_times[0] - anno["开始时间_s"])
            if total_checks:
                self.detail_rows.append(
                    {
                        "视频名称": video_name,
                        "维度": "身份识别",
                        "对象": person,
                        "事件类型": anno["事件类型"],
                        "位置": anno["发生位置"],
                        "结果": round(correct_checks / total_checks * 100, 2),
                        "附加信息": "窗口识别正确率(%)",
                    }
                )
                if correct_checks < total_checks:
                    self.issues.append(IssueRecord(video_name, anno["开始时间_s"], "身份识别错误", f"{person} 在 {anno['发生位置']} 识别不稳定"))

    def generate_bad_case_clips(self, video_name: str, video_path: str, sampled: List[Dict[str, object]]) -> None:
        # 将新增问题类型加入拦截名单，防止生成不需要的 3 秒切片
        ignored_issue_types = {"轨迹跳变", "ID串号", "误合并", "ID断裂", "行为漏报", "行为误报", "ID变化", "检测框断裂"}
        issues = [issue for issue in self.issues if issue.video_name == video_name and issue.issue_type not in ignored_issue_types]
        if not issues:
            return

        sample_lookup = {round(row["timestamp"], 3): row["detections"] for row in sampled}
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for issue in issues:
            start_ts = max(0.0, issue.timestamp - 1.0)
            end_ts = min(frame_count / fps, issue.timestamp + 2.0)
            out_path = os.path.join(
                self.bad_case_dir,
                f"{safe_name(issue.issue_type)}_{safe_name(os.path.splitext(video_name)[0])}_{issue.timestamp:05.1f}s.mp4",
            )
            writer = self.create_bad_case_writer(out_path, fps, width, height)
            written_frames = 0
            for frame_index in range(int(start_ts * fps), min(int(end_ts * fps) + 1, frame_count)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                ts = round(round((frame_index / fps) * self.analysis_fps) / self.analysis_fps, 3)
                detections = sample_lookup.get(ts, [])
                overlay = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # 反向查字典换英文，去掉了可能会显示中文乱码的人名
                    action_en = EN_ACTION_MAP.get(det.get('action', '正常'), 'unknown_action')
                    text = f"ID={det['target_id']} Action={action_en}"
                    cv2.putText(overlay, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 220), 2)
                
                # 异常类型转换为英文
                issue_type_en = EN_ISSUE_TYPE_MAP.get(issue.issue_type, 'Issue')
                cv2.putText(
                    overlay,
                    f"Issue: {issue_type_en}",
                    (20, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (30, 30, 30),
                    2,
                )
                self.write_bad_case_frame(writer, overlay)
                written_frames += 1
            self.close_bad_case_writer(writer)
            if written_frames == 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
                print(f"警告: bad case 切片写入异常，文件可能不可播放: {out_path}")
        cap.release()

    def generate_fn_videos(self, video_name: str, video_path: str, sampled: List[Dict[str, object]]) -> None:
        """为 False Negative 行为漏报生成全时段视频"""
        cases_for_video = [c for c in self.fn_cases if c["video_name"] == video_name]
        if not cases_for_video:
            return

        sample_lookup = {round(row["timestamp"], 3): row["detections"] for row in sampled}
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for case in cases_for_video:
            start_ts = case["start_s"]
            end_ts = case["end_s"]
            person = case["person"]
            action = case["action"]

            out_name = f"行为漏报_{safe_name(os.path.splitext(video_name)[0])}_{start_ts:05.1f}s_to_{end_ts:05.1f}s.mp4"
            out_path = os.path.join(self.fn_dir, out_name)
            writer = self.create_bad_case_writer(out_path, fps, width, height)

            start_frame = int(start_ts * fps)
            end_frame = min(int(end_ts * fps) + 1, frame_count)

            for frame_index in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                ts = round(round((frame_index / fps) * self.analysis_fps) / self.analysis_fps, 3)
                detections = sample_lookup.get(ts, [])
                overlay = frame.copy()
                
                # 遍历当帧所有人和动作
                for det in detections:
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # 动作名称英文映射
                    action_en = EN_ACTION_MAP.get(det.get('action', '正常'), 'unknown_action')
                    text = f"ID={det['target_id']} Action={action_en}"
                    cv2.putText(overlay, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 220), 2)
                
                # 底部信息英文映射
                target_action_en = EN_ACTION_MAP.get(action, 'unknown_action')
                cv2.putText(
                    overlay,
                    f"FN: Target '{target_action_en}' missing",
                    (20, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (30, 30, 30),
                    2,
                )
                self.write_bad_case_frame(writer, overlay)
            self.close_bad_case_writer(writer)
        cap.release()

    def generate_fp_images(self, video_name: str, video_path: str) -> None:
        """为 False Positive 行为误报生成单帧图片"""
        cases_for_video = [c for c in self.fp_cases if c["video_name"] == video_name]
        if not cases_for_video:
            return

        cap = cv2.VideoCapture(video_path)
        for case in cases_for_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, case["frame_index"])
            ret, frame = cap.read()
            if not ret:
                continue

            det = case["det"]
            ts = case["ts"]

            # 仅对发生误报的那个对象框线并标注
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 动作名称英文映射，简化 reason 为纯英文
            action_en = EN_ACTION_MAP.get(det.get('action', '正常'), 'unknown_action')
            text_lines = [
                f"ID: {det['target_id']}",
                f"Reported Action: {action_en}",
                f"Type: False Positive"
            ]

            y_offset = max(25, y1 - 10 - 20 * (len(text_lines) - 1))
            for i, line in enumerate(text_lines):
                cv2.putText(frame, line, (x1, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            out_name = f"行为误报_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s.jpg"
            out_path = os.path.join(self.fp_dir, out_name)
            cv2.imencode(".jpg", frame)[1].tofile(out_path)

        cap.release()

    def generate_localization_error_images(self, video_name: str, video_path: str) -> None:
        cases_for_video = [c for c in self.loc_error_cases if c["video_name"] == video_name]
        if not cases_for_video:
            return

        cap = cv2.VideoCapture(video_path)
        for case in cases_for_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, case["frame_index"])
            ret, frame = cap.read()
            if not ret:
                continue

            det = case["det"]
            gt = case["gt"]
            err = case["err"]
            ts = case["timestamp"]

            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            text_lines = [
                f"Error: {err:.1f} cm",
                f"Det XY: ({det['x']:.1f}, {det['y']:.1f})",
                f"GT XY: ({gt[0]:.1f}, {gt[1]:.1f})"
            ]

            y_offset = max(25, y1 - 10 - 20 * (len(text_lines) - 1))
            for i, line in enumerate(text_lines):
                cv2.putText(frame, line, (x1, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            out_name = f"loc_err_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s.jpg"
            out_path = os.path.join(self.loc_error_dir, out_name)
            cv2.imencode(".jpg", frame)[1].tofile(out_path)

        cap.release()

    def generate_id_change_images(self, video_name: str, video_path: str) -> None:
        """为 ID变化 生成对比图片并保存至 id_change 文件夹"""
        cases_for_video = [c for c in self.id_change_cases if c["video_name"] == video_name]
        if not cases_for_video:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
            
        def draw_box(img, det, info_texts):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            y_offset = max(25, y1 - 10 - 20 * (len(info_texts) - 1))
            for i, text in enumerate(info_texts):
                cv2.putText(img, text, (x1, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        def write_image_safe(filepath: str, img: np.ndarray) -> None:
            cv2.imencode(".jpg", img)[1].tofile(filepath)

        for c in cases_for_video:
            ts = c["ts"]
            if c["mode"] == "single":
                folder_name = f"ID变化_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s"
                folder_path = os.path.join(self.id_change_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, c["prev_frame_idx"])
                ret, frame = cap.read()
                if ret:
                    draw_box(frame, c["prev_det"], [f"Prev ID: {c['prev_det']['target_id']}"])
                    write_image_safe(os.path.join(folder_path, "prev.jpg"), frame)
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, c["curr_frame_idx"])
                ret, frame = cap.read()
                if ret:
                    draw_box(frame, c["curr_det"], [f"Curr ID: {c['curr_det']['target_id']}"])
                    write_image_safe(os.path.join(folder_path, "curr.jpg"), frame)
            else:
                # multi 模式
                cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame_index"])
                ret, frame = cap.read()
                if ret:
                    draw_box(frame, c["det"], [f"Current ID: {c['det']['target_id']}", f"Expected ID: {c['expected_id']}"])
                    out_name = f"ID变化_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s.jpg"
                    write_image_safe(os.path.join(self.id_change_dir, out_name), frame)
        
        cap.release()

    def generate_detect_lose_images(self, video_name: str, video_path: str) -> None:
        """为 检测框断裂 生成图片并保存至 detect_lose 下的专属文件夹"""
        cases_for_video = [c for c in self.detect_lose_cases if c["video_name"] == video_name]
        if not cases_for_video:
            return

        cap = cv2.VideoCapture(video_path)
        for case in cases_for_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, case["frame_index"])
            ret, frame = cap.read()
            if not ret:
                continue

            ts = case["ts"]
            det_count = case["det_count"]
            
            # 如果有部分框存活，也画出来
            for det in case["detections"]:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID:{det['target_id']}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            text_lines = [
                f"Type: Detect Lose",
                f"Expected: {case['expected_count']}, Found: {det_count}"
            ]

            for i, line in enumerate(text_lines):
                cv2.putText(frame, line, (20, 40 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            folder_name = f"检测框断裂_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s"
            folder_path = os.path.join(self.detect_lose_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            out_path = os.path.join(folder_path, "frame.jpg")
            cv2.imencode(".jpg", frame)[1].tofile(out_path)

        cap.release()

    def generate_tracking_error_images(self, video_name: str, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
            
        def draw_box(img, det, info_texts):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            y_offset = max(25, y1 - 10 - 20 * (len(info_texts) - 1))
            for i, text in enumerate(info_texts):
                cv2.putText(img, text, (x1, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        def write_image_safe(filepath: str, img: np.ndarray) -> None:
            cv2.imencode(".jpg", img)[1].tofile(filepath)

        # 1. 轨迹跳变 (Track Jump)
        tj_cases = [c for c in self.track_jump_cases if c["video_name"] == video_name]
        for c in tj_cases:
            ts = c["ts"]
            folder_name = f"轨迹跳变_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s"
            folder_path = os.path.join(self.track_jump_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["prev_frame_idx"])
            ret, frame = cap.read()
            if ret:
                draw_box(frame, c["prev_det"], [f"ID: {c['prev_det']['target_id']}", f"XY: ({c['prev_det']['x']:.1f}, {c['prev_det']['y']:.1f})"])
                write_image_safe(os.path.join(folder_path, "prev.jpg"), frame)
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["curr_frame_idx"])
            ret, frame = cap.read()
            if ret:
                draw_box(frame, c["curr_det"], [f"ID: {c['curr_det']['target_id']}", f"XY: ({c['curr_det']['x']:.1f}, {c['curr_det']['y']:.1f})"])
                write_image_safe(os.path.join(folder_path, "curr.jpg"), frame)

        # 2. ID 串号 (ID Switch)
        ids_cases = [c for c in self.id_switch_cases if c["video_name"] == video_name]
        for c in ids_cases:
            ts = c["ts"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame_index"])
            ret, frame = cap.read()
            if ret:
                draw_box(frame, c["det"], [f"Current ID: {c['det']['target_id']}", f"Expected ID: {c['expected_id']}"])
                out_name = f"ID串号_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s.jpg"
                write_image_safe(os.path.join(self.id_switch_dir, out_name), frame)
                
        # 3. 误合并 (False Merge)
        fm_cases = [c for c in self.false_merge_cases if c["video_name"] == video_name]
        for c in fm_cases:
            ts = c["ts"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame_index"])
            ret, frame = cap.read()
            if ret:
                draw_box(frame, c["det"], [f"ID: {c['det']['target_id']}", "False Merge"])
                out_name = f"误合并_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s.jpg"
                write_image_safe(os.path.join(self.false_merge_dir, out_name), frame)
                
        # 4. ID 断裂 (ID Break)
        idb_cases = [c for c in self.id_break_cases if c["video_name"] == video_name]
        for c in idb_cases:
            ts = c["ts"]
            folder_name = f"ID断裂_{safe_name(os.path.splitext(video_name)[0])}_{ts:05.1f}s"
            folder_path = os.path.join(self.id_break_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["prev_frame_idx"])
            ret, frame = cap.read()
            if ret:
                draw_box(frame, c["prev_det"], [f"Old ID: {c['prev_det']['target_id']}"])
                write_image_safe(os.path.join(folder_path, "prev.jpg"), frame)
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["curr_frame_idx"])
            ret, frame = cap.read()
            if ret:
                draw_box(frame, c["curr_det"], [f"New ID: {c['curr_det']['target_id']}"])
                write_image_safe(os.path.join(folder_path, "curr.jpg"), frame)
                
        cap.release()

    def write_reports(self) -> None:
        p50_error = float(np.percentile(self.localization_errors, 50)) if self.localization_errors else 0.0
        p90_error = float(np.percentile(self.localization_errors, 90)) if self.localization_errors else 0.0
        max_error = float(max(self.localization_errors)) if self.localization_errors else 0.0
        recall = self.behavior_tp / (self.behavior_tp + self.behavior_fn) * 100 if (self.behavior_tp + self.behavior_fn) else 0.0
        ownership = self.behavior_owner_correct / self.behavior_tp * 100 if self.behavior_tp else 0.0
        reg_acc = self.registered_id_correct / self.registered_id_total * 100 if self.registered_id_total else 0.0
        unknown_reject = self.unknown_reject_correct / self.unknown_reject_total * 100 if self.unknown_reject_total else 0.0
        avg_behavior_latency = float(np.mean(self.behavior_latency)) if self.behavior_latency else 0.0
        avg_identity_latency = float(np.mean(self.identity_latency)) if self.identity_latency else 0.0

        self.summary_rows = [
            {"指标维度": "定位精度", "指标名称": "P50误差(cm)", "结果数值": round(p50_error, 2)},
            {"指标维度": "定位精度", "指标名称": "P90误差(cm)", "结果数值": round(p90_error, 2)},
            {"指标维度": "定位精度", "指标名称": "Max误差(cm)", "结果数值": round(max_error, 2)},
            {"指标维度": "ID稳定性", "指标名称": "轨迹跳变次数", "结果数值": self.track_jump_count},
            {"指标维度": "ID稳定性", "指标名称": "ID串号次数", "结果数值": self.id_switch_count},
            {"指标维度": "ID稳定性", "指标名称": "误合并次数", "结果数值": self.false_merge_count},
            {"指标维度": "ID稳定性", "指标名称": "ID断裂次数", "结果数值": self.id_break_count},
            {"指标维度": "ID稳定性", "指标名称": "ID变化次数", "结果数值": self.id_change_count}, 
            {"指标维度": "检测稳定性", "指标名称": "检测框断裂帧数", "结果数值": self.detect_lose_count}, # 新增统计指标
            {"指标维度": "行为识别", "指标名称": "检出率(%)", "结果数值": round(recall, 2)},
            {"指标维度": "行为识别", "指标名称": "平均报警延迟(s)", "结果数值": round(avg_behavior_latency, 2)},
            {"指标维度": "行为识别", "指标名称": "归属准确率(%)", "结果数值": round(ownership, 2)},
            {"指标维度": "行为识别", "指标名称": "误报频次", "结果数值": self.behavior_fp},
            {"指标维度": "身份识别", "指标名称": "已注册识别准确率(%)", "结果数值": round(reg_acc, 2)},
            {"指标维度": "身份识别", "指标名称": "陌生人拒识率(%)", "结果数值": round(unknown_reject, 2)},
            {"指标维度": "身份识别", "指标名称": "身份确认延迟(s)", "结果数值": round(avg_identity_latency, 2)},
        ]

        summary_csv = os.path.join(self.output_dir, "eval_report.csv")
        detail_csv = os.path.join(self.output_dir, "eval_details.csv")
        issues_csv = os.path.join(self.output_dir, "issues.csv")
        self.write_csv(summary_csv, self.summary_rows)
        self.write_csv(detail_csv, self.detail_rows)
        self.write_csv(issues_csv, [issue.__dict__ for issue in self.issues])

        workbook_path = os.path.join(self.output_dir, "eval_report.xlsx")
        self.write_xlsx(
            workbook_path,
            {
                "summary": self.summary_rows,
                "details": self.detail_rows,
                "issues": [issue.__dict__ for issue in self.issues],
            },
        )

        print("评测完成。")
        print(f"汇总报告: {summary_csv}")
        print(f"明细结果: {detail_csv}")
        print(f"问题清单: {issues_csv}")
        print(f"Bad cases (行为漏报视频): {self.fn_dir}")
        print(f"Bad cases (行为误报截图): {self.fp_dir}")
        print(f"Bad cases (ID变化截图): {self.id_change_dir}")
        print(f"Bad cases (检测框断裂截图): {self.detect_lose_dir}")
        print(f"Bad cases (追踪类截图): {self.track_jump_dir}, {self.id_switch_dir}, etc.")

    def location_to_xy(self, location_name: str) -> Tuple[float, float]:
        item = self.location_config[location_name]
        return float(item["x"]), float(item["y"])

    def samples_in_window(
        self,
        sampled: List[Dict[str, object]],
        start: float,
        end: float,
    ) -> Iterable[Dict[str, object]]:
        return [row for row in sampled if start <= row["timestamp"] <= end]

    def load_annotations(self, anno_file: str) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        with open(anno_file, "r", encoding="gbk", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = dict(row)
                item["开始时间_abs"] = parse_abs_time(item["开始时间"])
                item["结束时间_abs"] = parse_abs_time(item["结束时间"])
                rows.append(item)
        return rows

    def unique_video_names(self) -> List[str]:
        seen = []
        for row in self.annotations:
            if row["视频名称"] not in seen:
                seen.append(row["视频名称"])
        return seen

    def write_csv(self, path: str, rows: List[Dict[str, object]]) -> None:
        headers = list(rows[0].keys()) if rows else ["empty"]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            if rows:
                writer.writerows(rows)

    def write_xlsx(self, path: str, sheets: Dict[str, List[Dict[str, object]]]) -> None:
        workbook = Workbook()
        first = True
        for sheet_name, rows in sheets.items():
            worksheet = workbook.active if first else workbook.create_sheet(title=sheet_name)
            worksheet.title = sheet_name
            first = False
            headers = list(rows[0].keys()) if rows else ["empty"]
            worksheet.append(headers)
            for row in rows:
                worksheet.append([row.get(header, "") for header in headers])
        workbook.save(path)

    def write_api_dump(self, video_name: str, sampled: List[Dict[str, object]]) -> None:
        out_path = os.path.join(self.api_dump_dir, f"{os.path.splitext(video_name)[0]}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in sampled:
                record = {
                    "video_name": video_name,
                    "timestamp": row["timestamp"],
                    "frame_index": row["frame_index"],
                    "normalized_detections": row["detections"],
                    "raw_response": row.get("raw_response"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def create_bad_case_writer(self, out_path: str, fps: float, width: int, height: int) -> Dict[str, object]:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {"kind": "ffmpeg", "process": process}
        except (FileNotFoundError, OSError):
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"无法创建 bad case 视频文件: {out_path}")
            return {"kind": "opencv", "writer": writer}

    def write_bad_case_frame(self, writer_info: Dict[str, object], frame: np.ndarray) -> None:
        if writer_info["kind"] == "ffmpeg":
            process = writer_info["process"]
            if process.stdin is not None:
                process.stdin.write(frame.tobytes())
            return
        writer_info["writer"].write(frame)

    def close_bad_case_writer(self, writer_info: Dict[str, object]) -> None:
        if writer_info["kind"] == "ffmpeg":
            process = writer_info["process"]
            if process.stdin is not None:
                process.stdin.close()
            process.wait()
            return
        writer_info["writer"].release()

    def find_by_target_id(self, detections: List[Dict[str, object]], target_id: Optional[str]) -> Optional[Dict[str, object]]:
        if not target_id:
            return None
        for det in detections:
            if det["target_id"] == target_id:
                return det
        return None

    def nearest_detection(self, detections: List[Dict[str, object]], gt: Tuple[float, float]) -> Optional[Dict[str, object]]:
        if not detections:
            return None
        nearest = min(detections, key=lambda det: calc_distance((det["x"], det["y"]), gt))
        if calc_distance((nearest["x"], nearest["y"]), gt) > self.match_distance_threshold * 2:
            return None
        return nearest

    def find_detection(
        self,
        detections: List[Dict[str, object]],
        target_id: Optional[str],
        gt: Tuple[float, float],
    ) -> Optional[Dict[str, object]]:
        by_id = self.find_by_target_id(detections, target_id)
        if by_id is not None:
            return by_id
        return self.nearest_detection(detections, gt)


def build_provider(args: argparse.Namespace) -> DetectionProvider:
    if args.mock_dir:
        return MockDetectionProvider(args.mock_dir)
    if args.api_url:
        return ApiDetectionProvider(
            args.api_url,
            timeout=args.api_timeout,
            camera_id=args.camera_id,
            enable_face_recognition=args.enable_face_recognition,
            enable_behavior_detection=args.enable_behavior_detection,
            enable_spatial_positioning=args.enable_spatial_positioning,
            enable_target_tracking=args.enable_target_tracking,
            world_coord_scale=args.world_coord_scale,
        )
    raise ValueError("必须至少提供 --mock_dir 或 --api_url 其中之一。")


def main() -> None:
    parser = argparse.ArgumentParser(description="视频分析自动化评测脚本")
    parser.add_argument("--video_dir", required=True, help="视频目录")
    parser.add_argument("--anno_file", required=True, help="annotations.csv 路径")
    parser.add_argument("--config", required=True, help="location_config.json 路径")
    parser.add_argument("--output_dir", default="output_results", help="结果输出目录")
    parser.add_argument("--mock_dir", help="模拟 API 输出目录")
    parser.add_argument("--api_url", help="真实分析系统 API 地址")
    parser.add_argument("--api_timeout", type=float, default=10.0, help="API 超时秒数")
    parser.add_argument("--camera_id", default="camera_01", help="真实分析系统请求所需的 camera_id")
    parser.add_argument("--enable_face_recognition", action="store_true", default=True, help="启用人脸识别")
    parser.add_argument("--disable_face_recognition", action="store_false", dest="enable_face_recognition", help="关闭人脸识别")
    parser.add_argument("--enable_behavior_detection", action="store_true", help="启用行为识别")
    parser.add_argument("--enable_spatial_positioning", action="store_true", default=True, help="启用空间定位")
    parser.add_argument("--disable_spatial_positioning", action="store_false", dest="enable_spatial_positioning", help="关闭空间定位")
    parser.add_argument("--enable_target_tracking", action="store_true", default=True, help="启用目标追踪")
    parser.add_argument("--disable_target_tracking", action="store_false", dest="enable_target_tracking", help="关闭目标追踪")
    parser.add_argument("--world_coord_scale", type=float, default=1.0, help="对 API 返回 world_coordinates 的缩放系数，例如毫米转厘米可设为 0.1")
    parser.add_argument("--analysis_fps", type=float, default=5.0, help="分析采样帧率")
    parser.add_argument("--jump_speed_threshold", type=float, default=5000.0, help="轨迹跳变速度阈值(mm/s)")
    parser.add_argument("--match_distance_threshold", type=float, default=800.0, help="点位匹配距离阈值(mm)")
    parser.add_argument("--merge_distance_threshold", type=float, default=1400.0, help="误合并判定半径(mm)")
    
    # [新增] 追踪模式参数
    parser.add_argument("--tracking_mode", choices=["single", "multi"], default="single", help="追踪模式: single(单人) 或 multi(多人)")
    
    args = parser.parse_args()

    provider = build_provider(args)
    evaluator = VideoEvaluator(
        video_dir=args.video_dir,
        anno_file=args.anno_file,
        config_file=args.config,
        output_dir=args.output_dir,
        provider=provider,
        analysis_fps=args.analysis_fps,
        jump_speed_threshold=args.jump_speed_threshold,
        match_distance_threshold=args.match_distance_threshold,
        merge_distance_threshold=args.merge_distance_threshold,
        tracking_mode=args.tracking_mode, # [新增] 传入追踪模式
    )
    evaluator.run()


if __name__ == "__main__":
    main()