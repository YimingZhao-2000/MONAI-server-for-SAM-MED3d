import os
import torch
import numpy as np
from monailabel.interfaces.app import MONAILabelApp
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monai.transforms import Compose

from utils.infer_utils import (
    sam_model_infer_with_user_prompt,
    get_subject_and_meta_info,
    data_preprocess,
    data_postprocess
)


class SAMMed3DInfer(BasicInferTask):
    def __init__(self, path, network=None, type="segmentation",
                 labels=["organ"], dimension=3, description="SAM-Med3D 3D Seg"):
        super().__init__(path=path, network=network, type=type,
                         labels=labels, dimension=dimension,
                         description=description)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import medim
        ckpt = os.getenv("SAM_CHECKPOINT_PATH", "ckpt/sam_med3d_turbo.pth")
        self.model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt)
        self.meta_info = None

    def pre_transforms(self, data=None):
        # 保留，但不重复执行 preprocess 操作
        return Compose([])

    def inferer(self):
        def _infer_fn(d):
            # 1️⃣ 数据预处理嵌入模型运行之前
            subject, meta_info = get_subject_and_meta_info(d["image_path"], None)
            img, _, meta = data_preprocess(
                subject, meta_info,
                category_index=1,
                target_spacing=(1.5,1.5,1.5),
                crop_size=128
            )
            self.meta_info = meta

            img = img.to(self.device)
            pts = d.get("points", None)
            lbs = d.get("labels", None)
            num_clicks = d.get("num_clicks", 1)
            user_points = torch.tensor([pts], device=self.device).float() if pts else None
            user_labels = torch.tensor([lbs], device=self.device).long() if lbs else None

            with torch.no_grad():
                mask, _ = sam_model_infer_with_user_prompt(
                    model=self.model,
                    roi_image=img,
                    roi_gt=None,
                    num_clicks=num_clicks,
                    user_points=user_points,
                    user_labels=user_labels
                )
            d["pred"] = mask.cpu().numpy()
            return d
        return _infer_fn

    def post_transforms(self, data=None):
        return Compose([])

    def writer(self, data, extension=".nii.gz", dtype=None):
        mask = data["pred"]
        # 2️⃣ 使用预处理保存的 meta_info 来复原空间
        if self.meta_info:
            mask = data_postprocess(mask, self.meta_info).astype(np.uint8)
        return mask, {}


class SAMMed3DApp(MONAILabelApp):
    def __init__(self, app_dir, studies, **kwargs):
        super().__init__(app_dir, studies, **kwargs)
        self.models = {
            "sam_med3d": SAMMed3DInfer(
                path=app_dir,
                network=None,
                type="segmentation",
                labels=["organ"],
                dimension=3,
                description="SAM-Med3D 3D Segmentation"
            )
        } 