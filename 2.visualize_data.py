from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from lerobot.datasets.utils import write_json, serialize_dict

dataset = LeRobotDataset('eco65_pnp', root='./demo_data') # 如需使用示例数据，改为 root='./demo_data_example'

import torch

class EpisodeSampler(torch.utils.data.Sampler):
    """
    单个 episode 的采样器
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = int(dataset.meta.episodes[episode_index]["dataset_from_index"])
        to_idx = int(dataset.meta.episodes[episode_index]["dataset_to_index"])
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

# 选择要可视化的 episode 编号
episode_index = 3

episode_sampler = EpisodeSampler(dataset, episode_index)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)


from mujoco_env.y_env import SimpleEnv
xml_path = './model/demo_scene.xml'
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

step = 0
iter_dataloader = iter(dataloader)
PnPEnv.reset()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 从数据集中读取动作
        data = next(iter_dataloader)
        if step == 0:
            # 根据数据集恢复物体初始位姿
            PnPEnv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0,3:])
        # 从数据集中获取动作
        action = data['action'].numpy()
        obs = PnPEnv.step(action[0])

        # 将数据集中的图像显示到 RGB 叠加层
        PnPEnv.rgb_agent = data['observation.image'][0].numpy()*255
        PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy()*255
        PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
        PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
        # 3×256×256 → 256×256×3
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))
        PnPEnv.render()
        step += 1

        if step == len(episode_sampler):
            # 回到开头重新播放
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0

PnPEnv.env.close_viewer()

stats = dataset.meta.stats
PATH = dataset.root / 'meta' / 'stats.json'
stats = serialize_dict(stats)

write_json(stats, PATH)
