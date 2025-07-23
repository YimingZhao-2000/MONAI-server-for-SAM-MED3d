---

# 🧠 MONAI Label 应用集成文档 — 推理/训练模块开发指南

本指南用于帮助你在 `MONAI Label` 框架中自定义交互式推理任务（如 DeepEdit / SAMMed3D），以支持 3D 医学图像的高效交互分割。

---

## 📁 目录结构（最低可运行示例）

```bash
my_app/
│
├── main.py                # 启动入口
├── config.py              # 定义 App 及注册任务
│
├── models/
│   └── sam_med3d/         # 模型文件夹，保存 weights、config、log
│
├── infer/
│   └── sam_med3d.py       # 自定义推理任务类，继承 BasicInferTask
│
├── train/                 # 可选：训练器定义
├── score/                 # 可选：打分器定义
├── strategy/              # 可选：主动学习策略
└── transforms/            # 自定义 transforms（如 RAS 方向处理、坐标补偿等）
```

---

## 🚀 一、部署与运行入口（`main.py` + `config.py`）

### ✨ `main.py`

```python
from monailabel.interfaces.app import main
from config import SAMMed3DApp

if __name__ == "__main__":
    main(SAMMed3DApp())
```

### ✨ `config.py`

```python
from monailabel.interfaces.app import MONAILabelApp
from infer.sam_med3d import SAMMed3DInferTask
import os

class SAMMed3DApp(MONAILabelApp):
    def __init__(self):
        super().__init__(
            app_dir=os.path.dirname(__file__),
            studies=os.path.join(os.path.dirname(__file__), "studies"),
            conf={"preload": "true"},
            name="SAMMed3D App",
            description="Interactive SAM-based segmentation",
        )

    def init_infers(self):
        return {
            "SAMMed3D": SAMMed3DInferTask(
                path=os.path.join(self.models_dir, "sam_med3d"),
                type=InferType.DEEPEDIT,
                labels={"organ": 1, "background": 0},
                spatial_size=(128, 128, 128),
                target_spacing=(1.0, 1.0, 1.0),
                number_intensity_ch=3,
                dimension=3,
                description="SAM + DeepEdit hybrid interactive segmentation"
            )
        }
```

---

## ⚙️ 二、自定义推理模块（`BasicInferTask`）

### ✅ 必须重写的接口：

| 方法名                           | 作用                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| `pre_transforms(self, data)`  | 输入数据预处理。必须使用 `AddGuidanceFromPointsDeepEditd` 等 transform 以从 3D Slicer guidance 中解析出点 |
| `post_transforms(self, data)` | 推理后处理，如 softmax/argmax/还原坐标等                                                          |
| `inferer(self, data)`         | 返回推理器实例，如 `SimpleInferer()`、`SlidingWindowInferer()`                                  |

### ✅ 建议重写的接口：

| 方法名                                | 作用                                           |
| ---------------------------------- | -------------------------------------------- |
| `run_inferer(self, inputs, model)` | 控制模型前向传播逻辑（如 SAM prompt 拼接、补偿坐标、显式 batch 构造） |
| `__init__()`                       | 指定推理模型、维度、标签映射、描述等参数                         |
| `run()`                            | 不建议重写，除非完全自定义执行逻辑                            |
| `work()`                           | 用于长期运行后台推理任务，例如实时标注、点扩展等                     |

---

## 🧩 三、推理流程详解

```python
class SAMMed3DInferTask(BasicInferTask):

    def pre_transforms(self, data):
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0),
            AddGuidanceFromPointsDeepEditd(
                ref_image="image", guidance="guidance", label_names=self.labels
            ),
            ResizeGuidanceMultipleLabelDeepEditd(guidance="guidance", ref_image="image"),
            AddGuidanceSignalDeepEditd(
                keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch
            ),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def inferer(self, data=None):
        return SimpleInferer()

    def run_inferer(self, inputs, model):
        logits = model(inputs)  # 可插入自定义 mask、prompt 等处理
        return logits

    def post_transforms(self, data):
        return [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image", config_labels=self.labels),
            GetCentroidsd(keys="pred", centroids_key="centroids")
        ]
```

---

## 📍 四、guidance 用户交互数据处理

### 3D Slicer 请求格式：

```json
{
  "image": "study1/image.nii.gz",
  "guidance": {
    "background": [],
    "organ": [[205, 12, 154], [133, 11, 126]]
  }
}
```

### 注意事项：

1. 原始坐标为 `(X, Y, Z)`，需补偿 crop 偏移（`meta_info["cropping_params_functional"]`）
2. `AddGuidanceFromPointsDeepEditd` 会自动生成 `guidance` tensor，可用于 `AddGuidanceSignalDeepEditd` 添加 mask channel
3. 如果使用自定义推理流程，可手动调用以下方法进行补偿：

```python
def compensate_user_points_for_crop(user_points, meta_info):
    d_crop, _, h_crop, _, w_crop, _ = meta_info["cropping_params_functional"]
    crop_offset = torch.tensor([[[w_crop, h_crop, d_crop]]], dtype=torch.float32)
    return user_points - crop_offset
```

---

## 🧪 五、可选模块说明（score/train/strategy）

### ✅ `strategy/`：主动学习策略模块

* 接口基类：`monailabel.interfaces.tasks.strategy.ActiveLearningStrategy`
* 应用场景：自动选择最有信息量的数据用于人工标注（如 uncertainty, random 等）
* 启用后将作用于 `/activelearning/next_sample`

### ✅ `train/`：训练任务模块

* 接口基类：`monailabel.interfaces.tasks.train.TrainTask`
* 自定义训练器，用于 `/train` 接口进行增量学习或微调
* 可集成 TensorBoard、Amp、DALI 等高级功能

### ✅ `score/`：评估任务模块

* 接口基类：`monailabel.interfaces.tasks.scoring.ScoringMethod`
* 用于评分模型性能或样本质量（用于主动学习选择、数据分析等）

---

## 📦 六、TaskConfig 可选封装（进阶）

虽然大多数项目不使用，但 `TaskConfig` 可用于集中式参数配置，主要包含：

* 模型结构定义
* transforms 注册
* loss 函数与优化器统一管理

```python
from monailabel.interfaces.config import TaskConfig

class MyTaskConfig(TaskConfig):
    def __init__(self):
        super().__init__()
        self.network = UNet(...)
        self.transforms_train = Compose([...])
        self.loss_function = DiceCELoss()
```

---


