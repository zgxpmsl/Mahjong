# Mahjong Logger

Mahjong Logger 提供了一个完整的流水线，用于在现有的 [Mahjong-YOLO](https://github.com/nikmomo/Mahjong-YOLO) 模型基础上进行数据标注、模型训练和日志评估。该项目聚焦贵阳捉鸡麻将的整局事件追踪，包含以下能力：

- **模型版本管理**：导入原始 YOLO11 权重或自训练模型，生成多版本模型库。
- **自动 / 半自动标注**：利用导入的模型对视频进行推理，一键生成标签并允许人工修正。
- **GUI**：通过 PySide6 提供桌面端界面，可导入视频、浏览帧、管理轨迹 ID、编辑标签、触发训练流程。
- **训练流水线**：基于 Ultralytics YOLO API 对数据进行增量训练，并记录训练产物与评估指标。
- **日志与状态机基础结构**：提供事件日志数据结构，便于后续实现规则推理与整局对比。

> **提示**：项目提供的是完整的代码框架，部分功能（如模型训练）依赖第三方库，需要在具备 GPU 的环境中运行。

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 准备数据目录

默认情况下，项目会在仓库根目录生成以下结构：

```
Mahjong/
├── data/
│   ├── raw_videos/         # 放置原始视频
│   └── annotations/        # GUI 保存的标注与日志
├── models/                 # 模型版本库
└── exports/                # 训练导出目录
```

可将原始贵阳捉鸡麻将视频放入 `data/raw_videos/` 供 GUI 导入。

### 3. 启动 GUI

```bash
mahjong-logger gui --root /path/to/project
```

GUI 提供以下工作流：

1. **Open Video**：选择原始视频，系统会创建一个新的标注 Session。
2. **Import Model**：导入 Mahjong-YOLO 或自定义 YOLO11 权重，系统会记录为一个新版本。
3. **Auto Label**：使用当前选中的模型对视频执行推理，自动填充每帧的牌面检测结果。
4. **人工修正**：通过表格修改 Track ID、牌面类别（使用默认牌面词表）。
5. **Save Session**：保存当前标注文件（包含帧级 bbox、事件日志占位字段）。
6. **Train**：选择基础模型与训练轮次，启动 Ultralytics YOLO 训练并将最佳模型保存到模型库。

### 4. 模型库管理

- 所有导入或训练生成的模型都会保存在 `models/` 下，每个版本包含权重文件与 `metadata.json`。
- GUI 在完成训练后会自动将最新模型设置为当前活跃模型，方便继续进行自动标注。

### 5. 事件日志

虽然 GUI 目前以帧级标注为主，但 `src/mahjong_logger/data/log_schema.py` 提供了事件日志的数据结构，后续可：

- 在 GUI 中扩展事件编辑界面，用于记录摸牌、出牌、碰、杠等动作。
- 使用 `pipeline/evaluation.py` 中的工具与人工日志进行对比，衡量事件重建的准确率。

### 6. 训练输出

训练完成后，Ultralytics 将在 `exports/<project>/<experiment>` 中生成完整日志、可视化以及最佳权重。项目会自动将最佳权重登记入模型库，便于继续迭代或导出部署。

## 目录说明

- `src/mahjong_logger/config.py`：全局路径与牌面类别配置。
- `src/mahjong_logger/data/annotation_store.py`：帧级标注、ID 持久化、YOLO 标签导出逻辑。
- `src/mahjong_logger/data/log_schema.py`：事件日志数据结构及序列化工具。
- `src/mahjong_logger/data/manager.py`：数据集管理、标注 Session 的加载 / 保存。
- `src/mahjong_logger/models/yolo_manager.py`：模型版本库与导入 / 导出逻辑。
- `src/mahjong_logger/pipeline/auto_label.py`：自动推理生成标签。
- `src/mahjong_logger/pipeline/training.py`：基于 Ultralytics 的训练封装。
- `src/mahjong_logger/pipeline/evaluation.py`：日志对比的简单评估工具。
- `src/mahjong_logger/gui/app.py`：PySide6 GUI 实现。
- `src/mahjong_logger/cli.py`：命令行入口。

## 后续扩展建议

- 集成玩家状态机与规则校验，将标注与事件日志联动，支持延迟确认与异常告警。
- 引入多目标跟踪与 ReID 模块，在 GUI 中可视化牌 ID 的全局轨迹。
- 扩展 GUI，使其支持事件时间线、日志对比、主动学习循环。
- 在 `pipeline/training.py` 中加入验证集划分、更多数据增强策略以及模型蒸馏。
- 基于模型库信息提供一键导出 ONNX / TensorRT / NCNN 推理包，便于边缘部署。

## 许可证

本项目示例代码以 MIT License 发布，具体条款请参阅 `LICENSE`（如适用）。
