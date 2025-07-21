import os
import torch
import numpy as np
from monailabel.interfaces.app import MONAILabelApp
from monailabel.tasks.infer.basic_infer import BasicInferTask

# 直接使用现有的推理函数
from utils.infer_utils import sam_model_infer_with_user_prompt, get_subject_and_meta_info, data_preprocess, data_postprocess


class SAMMed3DInfer(BasicInferTask):
    def __init__(self, model_dir, network=None, **kwargs):
        super().__init__(model_dir=model_dir, network=network, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 直接使用medim创建模型（与medim_val_dataset.py一致）
        import medim
        checkpoint_path = os.getenv("SAM_CHECKPOINT_PATH", "ckpt/sam_med3d_turbo.pth")
        self.model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=checkpoint_path)
        self.meta_info = None  # 保存meta_info用于postprocess

    def preprocess(self, request):
        # 直接使用现有的预处理逻辑
        image_path = request.get("image_path")
        if image_path:
            # 使用现有的数据加载函数
            subject, meta_info = get_subject_and_meta_info(image_path, None)
            # 使用data_preprocess确保ROI输入格式正确
            roi_image, _, meta_info = data_preprocess(
                subject, meta_info,
                category_index=1,  # 默认organ类别
                target_spacing=(1.5, 1.5, 1.5),
                crop_size=128
            )
            self.meta_info = meta_info  # 保存用于postprocess
            return {"image": roi_image}
        return request

    def infer(self, request):
        # 获取图像和点击信息
        image = request["image"]
        points = request.get("points", None)
        labels = request.get("labels", None)
        num_clicks = request.get("num_clicks", 1)
        
        # 转换用户输入的points/labels为tensor
        if points and labels:
            user_points = torch.tensor([points], dtype=torch.float, device=self.device)
            user_labels = torch.tensor([labels], dtype=torch.int64, device=self.device)
        else:
            user_points = user_labels = None
        
        # 使用新的推理函数，支持用户输入的prompt
        with torch.no_grad():
            mask, _ = sam_model_infer_with_user_prompt(
                model=self.model,
                roi_image=image,
                roi_gt=None,  # 推理时没有GT
                num_clicks=num_clicks,
                user_points=user_points,
                user_labels=user_labels
            )
        
        return mask

    def postprocess(self, mask):
        # 还原mask到原始空间
        if self.meta_info:
            final_mask = data_postprocess(mask, self.meta_info)
            return final_mask.astype(np.uint8)
        return mask


class SAMMed3DApp(MONAILabelApp):
    def __init__(self, app_dir, studies, **kwargs):
        super().__init__(app_dir, studies, **kwargs)
        
        # 注册推理任务
        self.models = {
            "sam_med3d": SAMMed3DInfer(
                model_dir=app_dir,
                network=None,
                description="SAM-Med3D 3D Medical Image Segmentation",
                labels=["organ", "tissue"],
                dimension=3,
            )
        } 