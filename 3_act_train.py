import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
import torchvision

device = torch.device("cuda")

# 离线训练步数
training_steps = 3000
log_freq = 100

# 从头训练时需指定两件事：
#   - 输入/输出形状：正确设置策略维度
#   - 数据集统计量：用于输入/输出的归一化和反归一化
dataset_metadata = LeRobotDataset("eco65_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image")
# 策略通过配置类初始化，除输入/输出特征外其余使用默认值
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size= 10, n_action_steps=10)
# 解析 delta_timestamps 以构建 action chunk 数据
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# 使用配置和数据集统计量实例化策略
policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.meta.stats)
policy.train()
policy.to(device)

from torchvision import transforms

class AddGaussianNoise(object):
    """
    为张量添加高斯噪声
    """
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 添加噪声：输入输出均为张量
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# 构建图像变换流水线：添加高斯噪声后裁剪至 [0,1]
transform = transforms.Compose([
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])


# 使用 delta_timestamps 和图像变换实例化数据集
dataset = LeRobotDataset("eco65_pnp", delta_timestamps=delta_timestamps, root='./demo_data', image_transforms=transform)

# 创建优化器和离线训练 DataLoader
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device.type != "gpu",
    drop_last=False,
)


# 训练循环
step = 0
done = False
while not done:
    for batch in dataloader:
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, _ = policy.forward(inp_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break


# 保存策略到磁盘
policy.save_pretrained('./ckpt/act_y')

import torch

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = int(dataset.meta.episodes[episode_index]["dataset_from_index"])
        to_idx = int(dataset.meta.episodes[episode_index]["dataset_to_index"])
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

policy.eval()
actions = []
gt_actions = []
images = []
episode_index = 0
episode_sampler = EpisodeSampler(dataset, episode_index)
test_dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=1,
    shuffle=False,
    pin_memory=device.type != "gpu",
    sampler=episode_sampler,
)
policy.reset()
for batch in test_dataloader:
    inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    action = policy.select_action(inp_batch)
    actions.append(action)
    gt_actions.append(inp_batch["action"][:,0,:])
    images.append(inp_batch["observation.image"])
actions = torch.cat(actions, dim=0)
gt_actions = torch.cat(gt_actions, dim=0)
print(f"Mean action error: {torch.mean(torch.abs(actions - gt_actions)).item():.3f}")

'''
绘制预测动作与真值对比：7 个子图对应 6 个关节角 + 1 个夹爪，
蓝线 (pred) 为策略预测值，橙线 (gt) 为遥操作真值，
越贴合说明模仿效果越好。
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

action_dim = 7
action_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']

fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))

for i in range(action_dim):
    axs[i].plot(actions[:, i].cpu().detach().numpy(), label="pred")
    axs[i].plot(gt_actions[:, i].cpu().detach().numpy(), label="gt")
    axs[i].set_ylabel(action_names[i])
    axs[i].legend(loc='upper right')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = f'./training_plot_{timestamp}.png'
plt.tight_layout()
plt.savefig(save_path, dpi=150)
print(f'Plot saved to {save_path}')

