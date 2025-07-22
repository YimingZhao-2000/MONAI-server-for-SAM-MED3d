import os
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferType

from sam_med3d_deepedit import SAMMed3DDeepEdit  # 假设你把刚才的 DeepEdit 类保存成这个文件


class SAMMed3DApp(MONAILabelApp):
    def __init__(self, app_dir, studies, **kwargs):
        super().__init__(app_dir, studies, **kwargs)

        # 注册模型 - 注意这里 type 设置为 InferType.DEEPEDIT
        infer_task = SAMMed3DDeepEdit(
            path=app_dir,
            network=None,  # 因为我们用的是 sam_model_infer_with_user_prompt, 自己控制了 model 初始化
            type=InferType.DEEPEDIT,
            labels=["organ"],  # 这里可以根据需要改 label 名称
            dimension=3,
            description="SAM‑Med3D DeepEdit App",
        )

        self.infers = {
            "sam_med3d_deepedit": infer_task
        }

        # Strategy 是可选的，这里不需要额外配置，除非需要 Active Learning 策略
        self.strategies = {}

        # 评分方法可选
        self.scoring_methods = {}

        # Train 任务可以不加（这里只用于 inference）

