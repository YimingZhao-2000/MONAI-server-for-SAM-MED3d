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
    def __init__(self, path, network=None, type="deepedit",
                 labels=["organ"], dimension=3, description="SAM-Med3D DeepEdit"):
        super().__init__(path=path, network=network, type=type,
                         labels=labels, dimension=dimension, description=description)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        import medim
        ckpt = os.getenv("SAM_CHECKPOINT_PATH", "ckpt/sam_med3d_turbo.pth")
        self.model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt)
        self.meta_info = None

    def pre_transforms(self, data=None):
        # 必须有定义，即使为空；否则 MONAI Label 0.8.x 在 pipeline 初始化会异常
        return Compose([])

    def run_inferer(self, data, device=None, **kwargs):
        # 获取 image_path 并运行自定义预处理逻辑
        image_path = data.get("image_path")
        assert image_path is not None, "Missing image_path in request!"

        subject, meta_info = get_subject_and_meta_info(image_path, None)
        roi_image, _, meta_info = data_preprocess(
            subject, meta_info,
            category_index=1,  # 默认 organ 类别，可改为传入
            target_spacing=(1.5, 1.5, 1.5),
            crop_size=128
        )
        self.meta_info = meta_info

        roi_image = roi_image.to(self.device)
        num_clicks = data.get("num_clicks", 1)
        user_points = data.get("points", None)
        user_labels = data.get("labels", None)

        if user_points and user_labels:
            user_points = torch.tensor([user_points], dtype=torch.float, device=self.device)
            user_labels = torch.tensor([user_labels], dtype=torch.int64, device=self.device)
        else:
            user_points = user_labels = None

        with torch.no_grad():
            mask, _ = sam_model_infer_with_user_prompt(
                model=self.model,
                roi_image=roi_image,
                roi_gt=None,  # 这里我们默认 inference，无 GT
                num_clicks=num_clicks,
                user_points=user_points,
                user_labels=user_labels
            )
        
        data["pred"] = mask.astype(np.uint8)
        return data

    def post_transforms(self, data=None):
        # 必须有定义，即使为空
        return Compose([])

    def writer(self, data, extension=".nii.gz", dtype=None):
        mask = data["pred"]
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
                type="deepedit",
                labels=["organ"],
                dimension=3,
                description="SAM-Med3D DeepEdit Model"
            )
        }
