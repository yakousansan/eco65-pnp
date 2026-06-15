"""
pi0 模型训练与部署脚本
基于 LeRobot pi0 预训练权重，使用本项目采集的 ECO65 数据集进行微调与部署。
"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision

device = 'cuda'

# ---------- 加载数据集 ----------
dataset_metadata = LeRobotDatasetMetadata("eco65_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image")  # 不使用腕部图像

# ---------- 初始化 pi0 策略 ----------
# pi0 是 VLA（Vision-Language-Action）模型，基于大规模机器人数据预训练
# 只需少量演示数据即可微调泛化
cfg = PI0Config(
    input_features=input_features,
    output_features=output_features,
    chunk_size=5,
    n_action_steps=5,
)
cfg.pretrained_path = 'lerobot/pi0'  # HuggingFace 预训练权重

# 从预训练检查点加载（需先运行训练）
# policy = PI0Policy.from_pretrained('./ckpt/pi0_eco65/checkpoints/last/pretrained_model',
#                                     config=cfg, dataset_stats=dataset_metadata.stats)
# 或直接从 HuggingFace Hub 加载已训练模型
# policy = PI0Policy.from_pretrained("your_username/eco65_pnp_pi0", config=cfg,
#                                     dataset_stats=dataset_metadata.stats)
# policy.to(device)

# ---------- 初始化 MuJoCo 环境 ----------
from mujoco_env.y_env import SimpleEnv
xml_path = './model/demo_scene.xml'
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

from torchvision import transforms

def get_default_transform(image_size: int = 224):
    """图像预处理：PIL → FloatTensor [0,1]"""
    return transforms.Compose([
        transforms.ToTensor(),
    ])

step = 0
PnPEnv.reset()
policy.reset()
policy.eval()
IMG_TRANSFORM = get_default_transform()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 检查任务是否完成
        success = PnPEnv.check_success()
        if success:
            print('Success')
            policy.reset()
            PnPEnv.reset()
            step = 0

        # 获取环境状态和图像
        state = PnPEnv.get_ee_pose()  # 末端位姿 [x,y,z,roll,pitch,yaw]
        image, wrist_image = PnPEnv.grab_image()
        image = Image.fromarray(image).resize((256, 256))
        image = IMG_TRANSFORM(image)
        wrist_image = Image.fromarray(wrist_image).resize((256, 256))
        wrist_image = IMG_TRANSFORM(wrist_image)

        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': ['Put mug cup on the plate'],
        }

        # 策略推理
        action = policy.select_action(data)
        action = action[0, :7].cpu().detach().numpy()

        # 执行动作
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1

        success = PnPEnv.check_success()
        if success:
            print('Success')
            break

# 推送到 HuggingFace Hub（可选）
# policy.push_to_hub(
#     repo_id='your_username/eco65_pnp_pi0',
#     commit_message='Add trained pi0 policy for ECO65 PnP task',
# )
