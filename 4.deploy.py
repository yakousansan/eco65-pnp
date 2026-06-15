from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.datasets.utils import write_json, serialize_dict
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision

# device = 'cuda'
device = 'cpu'
dataset_metadata = LeRobotDatasetMetadata("eco65_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image")
# 使用时间集成 (temporal ensemble) 平滑轨迹预测
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size= 10, n_action_steps=1, temporal_ensemble_coeff = 0.9)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# 从预训练检查点加载策略
policy = ACTPolicy.from_pretrained('./ckpt/act_y', config = cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)

from mujoco_env.y_env import SimpleEnv
xml_path = './model/demo_scene.xml'
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

step = 0
PnPEnv.reset()
policy.reset()
policy.eval()
save_image = True
img_transform = torchvision.transforms.ToTensor()
while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 检查任务是否完成
        success = PnPEnv.check_success()
        if success:
            print('Success')
            # 重置环境和动作队列
            policy.reset()
            PnPEnv.reset()
            step = 0
            save_image = False
        # 获取环境当前状态
        state = PnPEnv.get_ee_pose()
        # 获取环境当前图像
        image, wirst_image = PnPEnv.grab_image()
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = img_transform(image)
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = img_transform(wrist_image)
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': ['Put mug cup on the plate'],
            'timestamp': torch.tensor([step/20]).to(device)
        }
        # 策略推理选择动作
        action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()
        # 在仿真环境中执行动作
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1
        success = PnPEnv.check_success()
        if success:
            print('Success')
            break



