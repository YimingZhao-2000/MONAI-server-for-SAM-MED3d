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
        super().__init__(path=path,
                         network=network,
                         type=type,
                         labels=labels,
                         dimension=dimension,
                         description=description)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import medim
        ckpt = os.getenv("SAM_CHECKPOINT_PATH", "ckpt/sam_med3d_turbo.pth")
        self.model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt)
        self.meta_info = None

    def pre_transforms(self, data=None):
        # 可在此添加标准化、裁剪等MONAI transforms
        return Compose([])

    def preprocess(self, request):
        # 直接使用现有的预处理逻辑
        image_path = request.get("image_path")
        if image_path:
            subject, meta_info = get_subject_and_meta_info(image_path, None)
            roi_image, _, meta_info = data_preprocess(
                subject, meta_info,
                category_index=1,
                target_spacing=(1.5, 1.5, 1.5),
                crop_size=128
            )
            self.meta_info = meta_info
            return {"image": roi_image}
        return request

    def inferer(self, data=None):
        def _infer_fn(d):
            img = d["image"][None].to(self.device)
            pts = d.get("points", None)
            lbs = d.get("labels", None)
            num_clicks = d.get("num_clicks", 1)
            user_points = torch.tensor([pts], dtype=torch.float, device=self.device) if pts else None
            user_labels = torch.tensor([lbs], dtype=torch.int64, device=self.device) if lbs else None

            with torch.no_grad():
                mask, _ = sam_model_infer_with_user_prompt(
                    model=self.model,
                    roi_image=img,
                    roi_gt=None,
                    num_clicks=num_clicks,
                    user_points=user_points,
                    user_labels=user_labels
                )
            d["pred"] = mask
            return d
        return _infer_fn

    def post_transforms(self, data=None):
        # 可在此添加MONAI后处理transforms
        return Compose([])

    def postprocess(self, mask):
        # 还原mask到原始空间
        if self.meta_info:
            final_mask = data_postprocess(mask, self.meta_info)
            return final_mask.astype(np.uint8)
        return mask

    def writer(self, data=None, extension=".nii.gz", dtype=None):
        mask = data["pred"]
        meta = self.meta_info
        if meta:
            mask = data_postprocess(mask, meta).astype(np.uint8)
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