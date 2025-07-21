# MONAI-server-for-SAM-MED3d

本项目为 [SAM-Med3D](https://github.com/bowang-lab/SAM-Med3D) 的 MONAI Label 服务端适配，支持3D Slicer等客户端的交互式医学影像分割。

## 主要内容
- `config.py`：MONAI Label App 配置，集成 SAM-Med3D 推理逻辑
- `infer_utils.py`：推理与后处理工具，支持用户交互点输入
- `requirements.txt`：依赖包列表
- `start_server.sh`：一键启动 MONAI Label 服务脚本

## 快速开始

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
2. 准备模型权重
   ```bash
   export SAM_CHECKPOINT_PATH="ckpt/sam_med3d_turbo.pth"
   ```
3. 启动服务
   ```bash
   ./start_server.sh
   ```
4. 通过 3D Slicer MONAI Label 插件连接本服务，实现交互式分割。

## 依赖
- torch
- torchvision
- numpy
- nibabel
- monai-label
- torchio
- medim

## 说明
- 推理时支持 Slicer 客户端传入的点击点（points/labels），也支持无交互时自动中心点分割。
- 代码遵循奥卡姆剃刀原则，最大程度复用原有 SAM-Med3D 推理逻辑。

---
如有问题欢迎提 issue 或 PR！ 