"""
运行 4.deploy.py 的推理流程，同时将 MuJoCo 视窗画面保存为 MP4 视频。
输出: media/deploy_record_YYYYMMDD_HHMMSS.mp4
"""
import numpy as np
import torch
import torchvision
from PIL import Image
from datetime import datetime

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps

# ---------- 配置 ----------
FPS = 20               # 录像帧率
RECORD_DIR = "media"   # 输出目录
DEVICE = 'cuda'

# ---------- 加载策略 ----------
dataset_metadata = LeRobotDatasetMetadata("eco65_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image")

cfg = ACTConfig(input_features=input_features, output_features=output_features,
                chunk_size=10, n_action_steps=1, temporal_ensemble_coeff=0.9)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
policy = ACTPolicy.from_pretrained('./ckpt/act_y', config=cfg, dataset_stats=dataset_metadata.stats)
policy.to(DEVICE)
policy.eval()

# ---------- 初始化环境 ----------
from mujoco_env.y_env import SimpleEnv
xml_path = './model/demo_scene.xml'
env = SimpleEnv(xml_path, action_type='joint_angle')
env.reset()
policy.reset()

# ---------- 准备录像 ----------
import os
os.makedirs(RECORD_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
video_path = os.path.join(RECORD_DIR, f'deploy_record_{timestamp}.mp4')

# 使用 OpenCV VideoWriter
import cv2
img_transform = torchvision.transforms.ToTensor()

step = 0
frames = []
print("开始录制，关闭 MuJoCo 窗口或按 Esc 停止...")

while env.env.is_viewer_alive():
    env.step_env()
    if env.env.loop_every(HZ=FPS):
        success = env.check_success()
        if success:
            print(f'[{step}] Success — 重置环境')
            policy.reset()
            env.reset()

        # 推理
        state = env.get_ee_pose()
        image, wrist_image = env.grab_image()
        image_pt = img_transform(Image.fromarray(image).resize((256, 256)))
        wrist_pt = img_transform(Image.fromarray(wrist_image).resize((256, 256)))

        data = {
            'observation.state': torch.tensor([state]).to(DEVICE),
            'observation.image': image_pt.unsqueeze(0).to(DEVICE),
            'observation.wrist_image': wrist_pt.unsqueeze(0).to(DEVICE),
            'task': ['Put mug cup on the plate'],
            'timestamp': torch.tensor([step / FPS]).to(DEVICE),
        }
        action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()
        env.step(action)
        env.render()

        # ---- 抓取视窗画面 ----
        viewer_img = env.env.grab_image()  # (H, W, 3) uint8 BGR
        viewer_img = cv2.cvtColor(viewer_img, cv2.COLOR_RGB2BGR)
        frames.append(viewer_img)

        step += 1

# ---------- 写入视频 ----------
env.env.close_viewer()

if frames:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f'视频已保存: {video_path}  ({len(frames)} 帧, {w}x{h})')
else:
    print('未录制到任何帧')
