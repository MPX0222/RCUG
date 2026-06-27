# RCUG

RCUG 面向 **Class-Incremental Learning (CIL)**：在冻结 CLIP 视觉主干的前提下，为每个增量任务训练图像侧 Adapter，并通过 MoE 路由聚合多任务特征，结合 CLIP 文本特征进行零样本式分类。

---

## 项目结构

```
RCUG/
├── main.py                 # 训练 / 测试入口
├── config.py               # 命令行 + YAML 配置解析
├── data_manager.py         # 数据集管理与增量任务划分
├── ReplayBank.py           # 经验回放（memory_size > 0 时启用）
├── run.sh                  # 快速启动脚本
├── requirements.txt
├── methods/
│   ├── CLIP_task_adapter.py
│   └── Base.py
├── model/
│   ├── CLIP_task_adapter_Net.py
│   ├── CLIP_Base_Net.py
│   └── backbone/           # Adapter, clip, MedCLIP 等
├── config_yaml/CLIP_Adapter/
├── datasets/               # 数据集加载器 + class_descs
└── utils/
```

---

## 环境配置

```bash
cd /path/to/My_CL_Bank/RCUG
conda create -n rcug python=3.8 -y
conda activate rcug
pip install -r requirements.txt
pip install medmnist   # 仅 MedMNIST 需要
```

预训练权重默认放在 `$HOME/pretrained_model/`，在 yaml 的 `pretrained_path` 中配置（`main.py` 会自动拼接 `$HOME`）：

| 骨干 | `backbone` | 权重 |
|------|------------|------|
| CLIP | `CLIP` | `pretrained_model/CLIP_ViT-B-16.pt` |
| OpenCLIP | `OpenCLIP` | 含 `open_clip_pytorch_model.bin` 的目录 |
| MedCLIP | `MedCLIP` | 见 `model/backbone/MedCLIP/model.py` |

---

## 需要修改的路径

### 1. 预训练模型

yaml 中 `usual.pretrained_path`，运行时解析为 `$HOME/<pretrained_path>`。

### 2. 日志与 checkpoint

yaml 中 `basic.save_path`（默认 `../checkpoint&log`），实际目录：

```
<save_path>/CLIP_task_adapter/<version_name>/
```

### 3. 数据集路径

图像/原始数据路径在 `datasets/` 各文件的 `download_data()` 中配置，换环境后需按本机数据位置修改。**`class_descs` 已使用相对路径，一般无需改动。**

| 数据集 | yaml `dataset_name` | 修改文件 | 需改位置 |
|--------|---------------------|----------|----------|
| CIFAR-100 | `cifar100` | `datasets/CIFAR100.py` | 第 64–65 行，数据根目录 |
| ImageNet-R | `imagenet-r` | `datasets/ImageNet_R.py` | 第 53–54 行 train/test；第 60 行 `class_name.txt` |
| ImageNet-100 | `imagenet100` | `datasets/ImageNet100.py` | 第 67–68 行 train/val |
| CUB-200 | `CUB200` | `datasets/CUB200.py` | 第 66–67 行，数据集根目录与 images |
| Cars196 | `Cars196` | `datasets/Cars196.py` | 第 63 行根目录；第 68 行 `split_StanfordCars.json` |
| Skin40 | `Skin40` | `datasets/Skin40.py` | 第 55–57 行，train/val 列表与 images 目录 |
| Skin8 | `Skin8` | `datasets/Skin8.py` | 第 55–57 行，ISIC2019 根目录与 train/test 列表 |
| MedMNIST | `MedMNIST` | `datasets/MedMNIST.py` | 第 33–41 行，6 处 `root=`（硬编码绝对路径） |

`data_manager.py` 通过 `dataset_name` 选择加载器，须与 yaml 中名称一致。

Skin8 / MedMNIST 暂无预置 yaml，可复制现有配置并将 `dataset_name` 改为对应名称。


## 配置文件

位于 `config_yaml/CLIP_Adapter/`：

| 文件 | 数据集 |
|------|--------|
| `CLIP_adapter_cifar100.yaml` | CIFAR-100 |
| `CLIP_adapter_imagenet_r.yaml` | ImageNet-R |
| `CLIP_adapter_imagenet100.yaml` | ImageNet-100 |
| `CLIP_adapter_cub200.yaml` | CUB-200 |
| `CLIP_adapter_cars196.yaml` | Cars196 |
| `CLIP_adapter_skin40.yaml` | Skin40 |

主要字段：

```yaml
basic:
  method: "CLIP_task_adapter"
  increment_steps: [10, 10, ...]    # 每 task 新增类别数
  version_name: "..."               # 实验名，用于日志目录
  save_checkpoint: False

usual:
  dataset_name: "cifar100"
  backbone: "CLIP"
  pretrained_path: "pretrained_model/CLIP_ViT-B-16.pt"
  batch_size: 64
  epochs: 30
  memory_size: 0                    # >0 启用经验回放

special:
  prompt_template: "a photo of a {}."
  use_addi_desc: False
```

命令行可覆盖 yaml 参数：

```bash
python main.py --yaml_path=config_yaml/CLIP_Adapter/CLIP_adapter_cifar100.yaml --batch_size=32
```

---

## 训练

```bash
cd /path/to/My_CL_Bank/RCUG
conda activate rcug
CUDA_VISIBLE_DEVICES=0 python main.py \
  --yaml_path=config_yaml/CLIP_Adapter/CLIP_adapter_cifar100.yaml
```

或使用 `run.sh`（需先修改其中的 `device` 和 `config`）。

`main.py` 默认 `is_train = True`，按增量 task 依次：加载数据 → 训练 Adapter → 评估 → 可选保存 checkpoint。

日志：`../checkpoint&log/CLIP_task_adapter/<version_name>/<version_name>.log`

---

## 测试（仅评估）

1. yaml 中设置 `save_checkpoint: True` 生成 checkpoint，或确认已有：

   ```
   ../checkpoint&log/CLIP_task_adapter/<version_name>/checkpoint_task*.pkl
   ```

2. `main.py` 第 55 行改为 `is_train = False`

3. yaml 中 `version_name`、`increment_steps`、`dataset_name` 与训练时保持一致

4. 同样命令运行 `main.py`

断点续训：`is_train = True` 且 `resume = True`，存在 checkpoint 时跳过对应 task 的训练。

---

