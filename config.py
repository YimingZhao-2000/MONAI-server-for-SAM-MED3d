import os
import torch
import numpy as np
from monailabel.interfaces.app import MONAILabelApp
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monai.transforms import Compose
from monailabel.tasks.strategy.ete.deepedit_strategy import DeepEditStrategy

from utils.infer_utils import (
    sam_model_infer_with_user_prompt,
    get_subject_and_meta_info,
    data_preprocess,
    data_postprocess
)

class SAMMed3DInfer(BasicInferTask):
    def __init__(self, path, network=None, type="deepedit",
                 labels=["organ"], dimension=3, description="SAM‑Med3D DeepEdit"):
        super().__init__(
            path=path, network=network, type=type,
            labels=labels, dimension=dimension, description=description,
            preload=True  # 提前加载模型，可以加速首次推理
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import medim
        ckpt = os.getenv("SAM_CHECKPOINT_PATH", "ckpt/sam_med3d_turbo.pth")
        self.model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt)
        self.meta_info = None

    def pre_transforms(self, data=None):
        # 这里添加 MONAI Transform 链，如果你自己在 data_preprocess 已经做过，则可以返回[]
        return Compose([])

    def inferer(self):
        def _inf(data):
            subject, meta = get_subject_and_meta_info(data["image_path"], None)
            img, _, meta = data_preprocess(subject, meta, category_index=1,
                                           target_spacing=(1.5,)*3, crop_size=128)
            self.meta_info = meta
            img = img.to(self.device)
            pts = data.get("points", None)
            lbs = data.get("labels", None)
            num_clicks = data.get("num_clicks", 1)
            user_points = torch.tensor([pts], dtype=torch.float, device=self.device) if pts else None
            user_labels = torch.tensor([lbs], dtype=torch.int64, device=self.device) if lbs else None

            with torch.no_grad():
                mask, _ = sam_model_infer_with_user_prompt(
                    model=self.model, roi_image=img, roi_gt=None,
                    num_clicks=num_clicks,
                    user_points=user_points,
                    user_labels=user_labels)
            data["pred"] = mask  # numpy (D,H,W)
            return data
        return _inf

    def post_transforms(self, data=None):
        return Compose([])

    def inverse_transforms(self, data=None):
        return []  # 启用逆向恢复空间变换

    def writer(self, data, extension=".nii.gz", dtype=None):
        mask = data["pred"]
        if self.meta_info:
            mask = data_postprocess(mask, self.meta_info).astype(np.uint8)

        # 从 meta_info 获取原始文件路径
        img_path = self.meta_info.get("original_subject_path", None)
        if not img_path and "image_path" in data:
            img_path = data["image_path"]

        if not img_path:
            # fallback 到 app 目录
            base_dir = self.path
            fname = "prediction" + extension
        else:
            # 构造输出目录：原始图像同级的 predictionMONAI 文件夹
            data_dir = os.path.dirname(img_path)
            base_dir = os.path.join(data_dir, "predictionMONAI")
            os.makedirs(base_dir, exist_ok=True)
            fname = os.path.splitext(os.path.basename(img_path))[0] + extension

        out_fname = os.path.join(base_dir, fname)

        # 保存 NIfTI（或 sitk.WriteImage）
        import SimpleITK as sitk
        sitk_img = sitk.GetImageFromArray(mask)
        sitk.WriteImage(sitk_img, out_fname)

        result_json = {
            "label": out_fname,
            "latencies": data.get("latencies", {})
        }
        return out_fname, result_json


class SAMMed3DApp(MONAILabelApp):
    def __init__(self, app_dir, studies, **kwargs):
        super().__init__(app_dir, studies, **kwargs)
        infer = SAMMed3DInfer(path=app_dir)
        self.models = {"sam_med3d": infer}
        self.strategies = [
            DeepEditStrategy(name="SAM‑DeepEdit",
                             description="Interactive DeepEdit using SAM‑Med3D",
                             model=infer)
        ]




class SAMMed3DApp(MONAILabelApp):
    def __init__(self, app_dir, studies, **kwargs):
        super().__init__(app_dir, studies, **kwargs)
        infer = SAMMed3DInfer(path=app_dir)
        self.models = {"sam_med3d": infer}
        self.strategies = [
            DeepEditStrategy(name="SAM‑DeepEdit",
                             description="Interactive DeepEdit using SAM‑Med3D",
                             model=infer)
        ]