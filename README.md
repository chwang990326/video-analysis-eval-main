# 视频分析项目评测

这个仓库现在包含一套可直接运行的“视频分析系统自动化评测”工具链，按手册中的流程落地了三个部分：

1. `setup_test_env.py`
   生成标准评测工作区、模拟视频、点位配置、标注文件和 mock API 输出。
2. `run_eval.py`
   执行正式评测，支持 `--mock_dir` 离线演示模式，也支持 `--api_url` 对接真实视频分析系统。
3. `视频分析课题评测方案.md` / `视频分析系统自动化评测操作执行手册.md`
   作为指标和执行流程说明。

## 目录结构

执行 `python setup_test_env.py --overwrite` 后会生成：

```text
eval_workspace/
├── annotations.csv
├── config/
│   └── location_config.json
├── mock_api_outputs/
│   ├── manifest.json
│   ├── 剧本1_单人静态身份.json
│   ├── 剧本2_单人动态行为.json
│   ├── 剧本3_多人交叉轨迹.json
│   └── 剧本4_干扰动作误报.json
└── videos/
    ├── 剧本1_单人静态身份.mp4
    ├── 剧本2_单人动态行为.mp4
    ├── 剧本3_多人交叉轨迹.mp4
    └── 剧本4_干扰动作误报.mp4
```

## 依赖

```bash
pip install pandas numpy opencv-python openpyxl
```

## 快速开始

先生成模拟数据：

```bash
python setup_test_env.py --overwrite
```

再运行离线 mock 评测：

```bash
python run_eval.py \
  --video_dir eval_workspace/videos \
  --anno_file eval_workspace/annotations.csv \
  --config eval_workspace/config/location_config.json \
  --mock_dir eval_workspace/mock_api_outputs
```

默认会在当前目录生成 `output_results/`，里面包含：

1. `eval_report.csv` 和 `eval_report.xlsx`
2. `eval_details.csv`
3. `issues.csv`
4. `bad_cases/` 错误切片视频

## 对接真实系统 API

若要接入真实系统，将 `--mock_dir` 换成 `--api_url`：

```bash
python run_eval.py \
  --video_dir ./videos \
  --anno_file ./annotations.csv \
  --config ./config/location_config.json \
  --api_url http://127.0.0.1:8080/analyze
```

脚本会按采样帧率抽帧，并向接口发送如下 JSON：

```json
{
  "video_name": "剧本2_单人动态行为.mp4",
  "timestamp": "1970-01-01T00:00:10.200+00:00",
  "camera_id": "camera_01",
  "associated_camera_ids": [],
  "enable_face_recognition": true,
  "enable_behavior_detection": false,
  "enable_spatial_positioning": true,
  "enable_target_tracking": true,
  "image": "...base64 jpeg...",
  "image_base64": "...base64 jpeg..."
}
```

当前脚本已兼容你贴出的接口文档，会从响应里的 `data.persons` 读取：

- `person_id` 作为跟踪/身份字段
- `world_coordinates` 作为定位坐标
- `behavior_events` 作为行为识别结果
- `bounding_box` 作为可视化框

建议 `world_coordinates` 和 `location_config.json` 坐标统一使用毫米(mm)；若单位不一致，可通过 `--world_coord_scale` 进行换算统一。

接口返回建议格式：

```json
{
  "targets": [
    {
      "target_id": "201",
      "x": 320,
      "y": 300,
      "bbox": [280, 240, 360, 420],
      "action": "摔倒",
      "recognized_name": "李四"
    }
  ]
}
```

## 当前模拟数据包含的故障样例

为了方便演示 bad case 输出，mock 数据中故意注入了几类问题：

1. 张三在静态身份测试中有短暂识别错误
2. 李四的“摔倒”有延迟，“抽烟”漏报
3. 多人交叉场景中包含 ID 串号、误合并、ID 断裂和陌生人误识别
4. 干扰动作场景中包含异常行为误报

这样跑完评测后，你可以直接看到汇总指标和错误切片是否符合预期。
