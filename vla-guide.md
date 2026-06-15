# ECO65 模仿学习项目：从数据采集到模型部署

> 本文档详细说明基于 LeRobot + MuJoCo + ACT 的机器人模仿学习全流程。

---

## 1. 项目概述

本项目实现了一个完整的模仿学习流程：通过键盘遥操作控制 MuJoCo 仿真中的 ECO65 机械臂完成 **"将杯子放到盘子上"** 的任务，采集演示数据，训练 ACT 策略网络，最终部署模型让机器人自主完成任务。

### 核心组件

| 组件 | 作用 | 技术 |
|---|---|---|
| MuJoCo | 物理仿真引擎 | 6 自由度 ECO65 + PGC140 夹爪 + D435i 相机 |
| LeRobot | 数据集管理框架 | v3.0 格式，Parquet + MP4 存储 |
| ACT | 模仿学习策略 | Action Chunking with Transformers |
| IK 求解器 | 逆运动学 | 阻尼最小二乘法 (DLS) |

### 四个核心脚本

```
1.collect_data.py    →  遥操作采集数据
2.visualize_data.py  →  可视化已采集的数据
3.train.py           →  训练 ACT 策略
4.deploy.py          →  部署训练好的策略
```

---

## 2. LeRobot 数据集详解

### 2.1 什么是 LeRobot

LeRobot 是 HuggingFace 开源的机器人学习框架，核心功能包括：
- **统一的数据格式**：标准化机器人数据的存储和组织方式
- **数据集管理**：录制、保存、加载和分享演示数据
- **策略训练**：提供多种模仿学习算法的 PyTorch 实现
- **模型 Hub**：可以上传/下载预训练模型

### 2.2 数据集元信息 (info.json)

元信息文件位于 `demo_data/meta/info.json`，定义了数据集的全局属性。

```json
{
    "codebase_version": "v3.0",      // LeRobot 格式版本号
    "robot_type": "eco65_pgc140",    // 机器人类型标识
    "total_episodes": 0,             // 总 episode 数（录制过程动态更新）
    "total_frames": 0,               // 总帧数（录制过程动态更新）
    "total_tasks": 0,                // 不同任务数量
    "fps": 20,                       // 数据采样帧率
    "splits": {},                    // 训练/测试集划分
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": { ... }              // 特征定义（见下文）
}
```

#### 字段详解

| 字段 | 类型 | 含义 | 作用 |
|---|---|---|---|
| `codebase_version` | string | 数据格式版本 | 确保数据与代码版本兼容，v3.0 对应 lerobot ≥0.4 |
| `robot_type` | string | 机器人标识 | 用于区分不同机器人采集的数据，方便数据集共享和复用 |
| `total_episodes` | int | 演示总条数 | 每位演示者完成一次完整任务 = 1 个 episode |
| `total_frames` | int | 总图像帧数 | 所有 episode 的帧数之和 |
| `fps` | int | 数据采集帧率 | 每秒采集多少帧数据，20fps 意味着每 50ms 采集一帧 |
| `splits` | dict | 数据划分 | `{"train": "0:8", "test": "8:10"}` 表示前 8 个 episode 训练，后 2 个测试 |
| `data_path` | string | Parquet 存储路径模板 | 时序数据（关节角、末端位姿等）存储为 Parquet 文件 |
| `video_path` | string | 视频存储路径模板 | 图像帧被编码为 MP4 视频存储，节省空间 |

### 2.3 Features（特征定义）

Features 是数据集的核心，定义了每一帧数据包含哪些字段。分为两类：

#### 用户定义特征（采集时指定）

##### observation.image（Agent 视角图像）
```json
{
    "dtype": "image",
    "shape": [256, 256, 3],
    "names": ["height", "width", "channels"]
}
```

| 属性 | 值 | 含义 |
|---|---|---|
| `dtype` | `"image"` | 数据类型为图像 |
| `shape` | `[256, 256, 3]` | 高 256 × 宽 256 × RGB 三通道 |
| `names` | `["height", "width", "channels"]` | 各维度的语义标签 |

**来源**：MuJoCo 中的 `agentview` 相机，安装在仿真场景中的固定第三方视角。提供环境和机器人的全局视图。

**作用**：让策略网络获得对场景的整体感知，知道物体位置、机械臂状态等全局信息。

---

##### observation.wrist_image（腕部相机图像）
```json
{
    "dtype": "image",
    "shape": [256, 256, 3],
    "names": ["height", "width", "channel"]
}
```

**来源**：MuJoCo 中的 `d435i_rgb` 相机，安装在机械臂末端（link_6）的 D435i 深度相机 RGB 输出。

**作用**：提供末端执行器的近距离视角，让策略更精确地知道夹爪与物体的相对位置关系。**注意**：在训练时此图像被排除（`input_features.pop("observation.wrist_image")`），仅作备用。

---

##### observation.state（机器人状态）
```json
{
    "dtype": "float32",
    "shape": [6],
    "names": ["state"]
}
```

**内容**：`[x, y, z, roll, pitch, yaw]` —— 末端执行器在基坐标系下的 6 维位姿。

| 维度 | 含义 | 单位 | 作用 |
|---|---|---|---|
| `x` | 末端 X 位置 | 米 (m) | 末端在世界坐标系的 X 坐标 |
| `y` | 末端 Y 位置 | 米 (m) | 末端在世界坐标系的 Y 坐标 |
| `z` | 末端 Z 位置 | 米 (m) | 末端在世界坐标系的 Z 坐标 |
| `roll` | 绕 X 轴旋转 | 弧度 (rad) | 末端姿态的滚转角 |
| `pitch` | 绕 Y 轴旋转 | 弧度 (rad) | 末端姿态的俯仰角 |
| `yaw` | 绕 Z 轴旋转 | 弧度 (rad) | 末端姿态的偏航角 |

**来源**：`env.get_ee_pose()` 方法，从 MuJoCo 中读取 `link_6` 的位置和旋转矩阵，转换为 RPY 角。

**作用**：提供机器人当前姿态的数值信息，配合视觉信息让策略网络做更精确的运动规划。

---

##### action（动作）
```json
{
    "dtype": "float32",
    "shape": [7],
    "names": ["action"]
}
```

**内容**：`[joint_1, joint_2, ..., joint_6, gripper]` —— 7 维动作向量。

| 维度 | 含义 | 范围 |
|---|---|---|
| `action[0]` | joint_1 角度 | ECO65 关节 1（肩部偏转） |
| `action[1]` | joint_2 角度 | ECO65 关节 2（肩部俯仰） |
| `action[2]` | joint_3 角度 | ECO65 关节 3（肘部俯仰） |
| `action[3]` | joint_4 角度 | ECO65 关节 4（腕部旋转） |
| `action[4]` | joint_5 角度 | ECO65 关节 5（腕部俯仰） |
| `action[5]` | joint_6 角度 | ECO65 关节 6（法兰旋转） |
| `action[6]` | 夹爪状态 | `0.0` = 闭合, `1.0` = 张开 |

**来源**：采集模式下由 `PnPEnv.step(action)` 返回，是操作者遥操作产生的实际动作。`state_type='joint_angle'` 时返回关节角 + 夹爪状态。

**作用**：这是策略网络的**预测目标**——训练后，网络根据 observation 预测应该输出什么 action。

---

##### obj_init（物体初始位姿）
```json
{
    "dtype": "float32",
    "shape": [6],
    "names": ["obj_init"]
}
```

**内容**：`[mug_x, mug_y, mug_z, plate_x, plate_y, plate_z]` —— 杯子和盘子初始位置。

**作用**：记录每个 episode 的物体初始位置。在可视化回放时用来复现相同的场景布局。**训练中不使用此特征**。

---

#### 自动添加特征（由 LeRobot 框架管理）

| 字段 | 类型 | 形状 | 含义 |
|---|---|---|---|
| `timestamp` | float32 | [1] | 时间戳（秒），= frame_index / fps |
| `frame_index` | int64 | [1] | 帧索引，从 0 开始递增 |
| `episode_index` | int64 | [1] | episode 编号，从 0 开始 |
| `index` | int64 | [1] | 全局帧索引，跨 episode 连续 |
| `task_index` | int64 | [1] | 任务索引，多任务场景下区分不同任务 |
| `task` | string | - | 任务描述文本（如 "Put mug cup on the plate"） |

这些字段由 `LeRobotDataset.add_frame()` 方法自动添加，无需手动指定。

### 2.4 数据存储结构

```
demo_data/
├── meta/
│   ├── info.json          # 数据集元信息
│   ├── episodes.jsonl     # 每行一个 episode 的描述
│   ├── tasks.jsonl        # 任务描述
│   └── stats.json         # 数据统计（均值、标准差），用于归一化
├── data/
│   └── chunk-000/
│       ├── file-000.parquet  # 时序数据（关节角、位姿等）
│       └── file-001.parquet
└── videos/
    ├── observation.image/
    │   └── chunk-000/
    │       └── file-000.mp4  # 第三人称视角视频
    └── observation.wrist_image/
        └── chunk-000/
            └── file-000.mp4  # 腕部相机视频
```

- **Parquet 文件**：存储除图像外的所有数值数据（state、action 等），每 1000 帧一个文件
- **MP4 视频**：图像帧被编码为视频，大幅节省磁盘空间
- **stats.json**：通过 `2.visualize_data.py` 计算生成，包含每个特征的均值和标准差，供训练归一化使用

### 2.5 数据采集流程（1.collect_data.py）

```
┌─────────────────┐
│  初始化环境      │  SimpleEnv(xml_path, ...)
│  (MuJoCo 场景)  │  加载 ECO65 + PGC140 + D435i + 桌面 + 物体
└────────┬────────┘
         ▼
┌─────────────────┐
│  创建数据集      │  LeRobotDataset.create(...)
│  (定义 features) │  指定 observation.state, action 等的 shape 和 dtype
└────────┬────────┘
         ▼
┌─────────────────┐      循环 20Hz
│  主循环          │ ← ─ ─ ─ ─ ─ ─ ─ ─ ┐
│                  │                      │
│  teleop_robot()  │  读取键盘输入        │
│       ↓          │  WASD=移动, QE=旋转  │
│  生成 delta_eef  │  Space=夹爪开合     │
│       ↓          │                      │
│  solve_ik()      │  逆运动学求解         │
│       ↓          │  末端位姿 → 关节角   │
│  step(action)    │  执行动作             │
│       ↓          │                      │
│  grab_image()    │  获取两个相机图像     │
│       ↓          │                      │
│  add_frame()     │  写入缓冲区           │
│       ↓          │                      │
│  check_success() │  任务完成？→ save    │
└──────────────────┘ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
         ▼
┌─────────────────┐
│  finalize()      │  编码视频, 写入磁盘
└─────────────────┘
```

#### 键盘遥操作映射

遥操作使用末端执行器增量控制（`eef_pose` 模式）：

| 按键 | 动作 | 方向 |
|---|---|---|
| `W` / `S` | 末端前进/后退 | 世界 X 轴，步长 0.007m |
| `A` / `D` | 末端左移/右移 | 世界 Y 轴，步长 0.007m |
| `R` / `F` | 末端上升/下降 | 世界 Z 轴，步长 0.007m |
| `Q` / `E` | 绕 Z 轴旋转 | 滚转 (roll)，步长 ~1.7°/次 |
| `↑` / `↓` | 绕 X 轴旋转 | 俯仰 (pitch)，步长 ~1.7°/次 |
| `←` / `→` | 绕 Y 轴旋转 | 偏航 (yaw)，步长 ~1.7°/次 |
| `Space` | 夹爪开合 | 0 ↔ 1 切换 |
| `Z` | 重置 | 丢弃当前 episode，重新开始 |

#### 每帧记录的数据

调用 `add_frame()` 时，以下数据被打包为一帧：

```python
dataset.add_frame({
    "observation.image":       agent_image,    # (256, 256, 3) 第三人称视角
    "observation.wrist_image": wrist_image,    # (256, 256, 3) 手腕视角
    "observation.state":       ee_pose,        # [6] 末端位姿
    "action":                  joint_q,        # [7] 关节角 + 夹爪
    "obj_init":                obj_init_pose,  # [6] 物体初始位置
    "task":                    TASK_NAME,      # 任务描述字符串
})
```

---

## 3. 训练详解

### 3.1 ACT 算法概述

**ACT（Action Chunking with Transformers）** 是一种基于 Transformer 的模仿学习算法，核心思想是：

> 不是预测单步动作，而是**一次性预测未来多个时间步的动作序列**（chunk），从而产生更平滑、更稳定的运动轨迹。

#### 为什么需要 Action Chunking？

- **克服误差累积**：单步预测的小误差会在执行过程中累积放大，chunk 预测隐式地为每个动作提供了上下文约束
- **处理时延**：真实机器人执行有延迟，提前预测 N 步可以让执行更流畅
- **学习运动原语**：chunk 天然捕捉了运动中的时序相关性（如"伸手→抓取→提起"是一个连续的动作模式）

### 3.2 算法架构

```
输入                          输出
┌──────────────┐             ┌──────────────────┐
│ image (3×256)│──┐          │  action chunk    │
│   ResNet18   │  │   ┌────┐ │  [10, 7]         │
│   ─────────  │──┼──▶│    │ │  t₀: [j1..j6, g] │
│  img_feat(512)│  │  │ACT │ │  t₁: [j1..j6, g] │
└──────────────┘  │  │    │ │  ...              │
                  │  │Tran│ │  t₉: [j1..j6, g] │
┌──────────────┐  │  │sfo─├▶└──────────────────┘
│ state [6]    │──┤  │rmer│
│  (x,y,z,     │  │  │Enco│
│   r,p,y)     │  │  │der─│
└──────────────┘  │  │Deco │
                  │  │der │
         ┌──────┐ │  │    │
         │ VAE  │ │  │    │
         │ (可选)│─┘  │    │
         └──────┘    └────┘
```

#### 网络组件

| 模块 | 作用 | 配置参数 |
|---|---|---|
| **Vision Backbone (ResNet18)** | 将 256×256 图像编码为 512 维特征向量 | 预训练 ImageNet 权重 |
| **State Encoder** | 将 6 维末端位姿映射到嵌入空间 | 线性投影 |
| **VAE Encoder**（可选） | 学习动作的潜在表示，提供风格化能力 | latent_dim=32, 4 层 |
| **Transformer Encoder** | 对 observation 特征序列做自注意力 | 4 层, dim_model=512, 8 heads |
| **Transformer Decoder** | 自回归生成动作 chunk | 1 层, 512 维, 8 heads |
| **VAE Decoder**（可选） | 从潜在编码解码动作 | 对称结构 |

#### 关键配置参数

从 `ckpt/act_y/config.json` 和 `3.train.py`：

| 参数 | 值 | 含义 |
|---|---|---|
| `chunk_size` | 10 | 每次预测 10 步连续动作 |
| `n_action_steps` | 10 | 每次执行 10 步（离线训练）/ 1 步（在线部署） |
| `n_obs_steps` | 1 | 使用 1 帧观测历史 |
| `dim_model` | 512 | Transformer 隐藏维度 |
| `n_heads` | 8 | 多头注意力头数 |
| `n_encoder_layers` | 4 | Transformer 编码器层数 |
| `n_decoder_layers` | 1 | Transformer 解码器层数 |
| `dim_feedforward` | 3200 | FFN 中间维度 |
| `dropout` | 0.1 | Dropout 比例 |
| `kl_weight` | 10.0 | VAE KL 散度权重 |
| `latent_dim` | 32 | VAE 潜在空间维度 |
| `vision_backbone` | resnet18 | 视觉特征提取器 |

### 3.3 训练流程（3.train.py）

#### Step 1: 加载数据集元信息

```python
dataset_metadata = LeRobotDataset("eco65_pnp", root='./demo_data')
```

加载 `info.json`、`episodes.jsonl`、`stats.json`，获取特征定义和数据统计信息。

#### Step 2: 特征分类

```python
features = dataset_to_policy_features(dataset_metadata.features)
# 按 FeatureType 分类
output_features = {ACTION}              # action
input_features = {VISUAL, STATE}        # observation.image, observation.state
input_features.pop("observation.wrist_image")  # 排除手腕图像
```

**为什么排除手腕图像？** 因为对于这个任务，agentview 已经提供了足够的全局视觉信息，增加手腕图像会增加计算开销而收益有限。如果后续需要更精细的抓取操作，可以保留。

#### Step 3: 构建策略配置

```python
cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    chunk_size=10,
    n_action_steps=10       # 训练时一步执行 10 步预测
)
```

#### Step 4: 构建 Action Chunk 数据集

```python
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
dataset = LeRobotDataset(..., delta_timestamps=delta_timestamps)
```

`delta_timestamps` 定义了训练时需要的时间偏移：
- `action` → `[0.0, 0.05, 0.10, ..., 0.45]`（10 步，每步 0.05s = 1/20fps）
- `observation.state` → `[0.0]`（当前帧状态）
- `observation.image` → `[0.0]`（当前帧图像）

这样每个样本的 action 是一个连续的 10 步轨迹片段。

#### Step 5: 数据增强

```python
transform = transforms.Compose([
    AddGaussianNoise(mean=0., std=0.02),  # 添加高斯噪声
    transforms.Lambda(lambda x: x.clamp(0, 1))
])
```

对**图像**添加高斯噪声（std=0.02），增强策略的鲁棒性。注意：这是直接对图像张量（范围 [0,1]）加噪，而非常见的像素级噪声。

#### Step 6: 训练循环

```python
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch in dataloader:
    loss, _ = policy.forward(inp_batch)  # 前向传播
    loss.backward()                       # 反向传播
    optimizer.step()                      # 更新参数
    optimizer.zero_grad()
```

**损失函数**：
- 如果启用 VAE：`loss = L1(action_pred, action_gt) + kl_weight * KL(q(z|action_gt) || p(z|obs))`
  - 第一项是动作重建损失
  - 第二项是 VAE 正则化项，防止潜在空间崩塌
- 如果不用 VAE：`loss = L1(action_pred, action_gt)` 直接回归

**优化器**：Adam，学习率 1e-4，训练 3000 步

#### Step 7: 保存模型

```python
policy.save_pretrained('./ckpt/act_y')
```

保存内容：
```
ckpt/act_y/
├── config.json       # 策略配置（架构参数）
├── model.safetensors # 模型权重（安全格式）
└── ...
```

---

## 4. 模型部署详解

### 4.1 部署流程（4.deploy.py）

```
┌─────────────────┐
│  加载模型        │  ACTPolicy.from_pretrained('./ckpt/act_y')
│  设置 eval 模式  │  policy.eval()
└────────┬────────┘
         ▼
┌─────────────────┐
│  初始化环境      │  SimpleEnv(xml_path, action_type='joint_angle')
│  reset(seed=0)  │  固定种子保证可复现
└────────┬────────┘
         ▼
┌─────────────────┐      循环 20Hz
│  主循环          │ ← ─ ─ ─ ─ ─ ─ ─ ─ ┐
│                  │                      │
│  grab_image()    │  获取当前图像        │
│  get_ee_pose()   │  获取末端位姿        │
│       ↓          │                      │
│  构建输入        │  {                   │
│    observation.  │    'state': [x,y,z,  │
│      state       │       r,p,y],        │
│    observation.  │    'image': tensor,  │
│      image       │    'wrist_image': t, │
│  }               │    'task': "...",    │
│       ↓          │    'timestamp': ts   │
│  policy.select   │  }                   │
│    _action()     │    → 推理            │
│       ↓          │                      │
│  得到 action     │  [7] 关节角+夹爪    │
│       ↓          │                      │
│  step(action)    │  在仿真中执行        │
│       ↓          │                      │
│  check_success() │  任务完成？→ reset   │
└──────────────────┘ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

### 4.2 关键差异：训练 vs 部署

| 方面 | 训练 (3.train.py) | 部署 (4.deploy.py) |
|---|---|---|
| **n_action_steps** | 10（一次预测 10 步，离线计算 loss） | 1（每次只执行 1 步，实时交互） |
| **temporal_ensemble_coeff** | None（不做时间集成） | 0.9（平滑预测） |
| **action 模式** | 批量预测，计算 loss | 逐帧预测，执行动作 |
| **稳定性** | 无特殊处理 | `policy.reset()` 清理历史状态 |

### 4.3 时间集成（Temporal Ensemble）

部署时启用 `temporal_ensemble_coeff=0.9`，这是 ACT 论文的核心技巧：

```
新预测的动作 = 当前时刻模型预测 × (1-α) + 上一时刻集成结果 × α
              where α = 0.9
```

**效果**：
- 同一个 chunk 内的相邻帧预测结果被指数加权平均
- 消除相邻预测之间的抖动，使轨迹更平滑
- α 越大，平滑效果越强，但响应越慢

### 4.4 动作执行流程

```
帧 0: policy.select_action(obs_0) → 预测 [a₀, a₁, a₂, ..., a₉]（10步）
       ← 只执行 a₀

帧 1: policy.select_action(obs_1) → 预测 [a₀', a₁', ..., a₉']
       ← 只执行 a₀'（但已用 temporal ensemble 平滑过）

帧 2: ...
```

由于 `n_action_steps=1`，每帧只取 chunk 的第一个动作执行，下一帧重新推理。时间集成确保相邻帧的动作不会跳变。

### 4.5 成功判定

```python
def check_success(self):
    # 1. 杯子和盘子水平距离 < 10cm
    dist_xy < 0.1
    
    # 2. 杯子在盘子上方（或下方不超过盘子高度）
    |mug_z - plate_z| < 0.6
    
    # 3. 夹爪已张开（释放杯子）
    finger1_joint < 0.1  (即接近最小开度)
    
    # 4. 机械臂已抬起
    link_6_z > 0.9
```

---

## 5. 完整工作流程总结

```
第 1 步：采集数据
──────────────────────────────────────────────
python 1.collect_data.py
  → 启动 MuJoCo 仿真窗口
  → 键盘遥操作完成 "将杯子放到盘子上"
  → 每完成一次 → 自动保存一个 episode
  → 关闭窗口 → 编码视频 → finalize


第 2 步：可视化验证
──────────────────────────────────────────────
python 2.visualize_data.py
  → 回放采集的 episode
  → 验证数据质量（动作是否平滑、物体位置是否正确）
  → 自动计算数据统计（均值/标准差）→ stats.json


第 3 步：训练策略
──────────────────────────────────────────────
python 3.train.py
  → 加载数据集 + stats
  → 初始化 ACT 网络（ResNet18 + Transformer + VAE）
  → 3000 步训练（Adam, lr=1e-4, batch=64）
  → 在 GPU 上约需 5-15 分钟（取决于数据量）
  → 保存模型到 ckpt/act_y/


第 4 步：部署测试
──────────────────────────────────────────────
python 4.deploy.py
  → 加载训练好的模型
  → 在 MuJoCo 中运行
  → 策略根据视觉 + 状态输入自主决策
  → 观察成功率 → 如果不好 → 回到第 1 步采集更多 demo
```

### 数据量与效果的关系

| 演示数量 | 预期效果 |
|---|---|
| 1-5 条 | 可能过拟合，仅熟悉特定物体位置 |
| 10-20 条 | 开始泛化，不同初始位置有一定成功率 |
| 50+ 条 | 较好的泛化能力 |
| 100+ 条 | 配合数据增强，可处理未见过的物体位置 |

---

## 6. 附录

### A. 动作空间说明

本项目的 `state_type='joint_angle'` 模式使用**关节空间动作**：

- 动作 = 6 个关节的目标角度 + 1 个夹爪命令
- 优点：直接、简单，不需要 IK 求解（部署时）
- 缺点：不如末端空间直观

如果改为 `action_type='eef_pose'`，则动作变成末端位姿增量，需要 IK 求解器将位姿转换为关节角。遥操作时实际用的是 eef_pose 模式（键盘控制的是末端增量），但数据采集使用 delta_joint_angle 作为 state_type。

### B. 归一化

所有特征使用 `MEAN_STD` 归一化：
- 训练时：(value - mean) / std
- 推理时的模型输出：反归一化得到实际值

stats 由 `2.visualize_data.py` 计算并保存到 `stats.json`。

### C. 关键文件索引

| 文件 | 作用 |
|---|---|
| `mujoco_env/y_env.py` | 环境封装，定义 state/action 空间、遥操作、IK |
| `mujoco_env/mujoco_parser.py` | MuJoCo 底层接口，XML 加载、渲染、相机 |
| `mujoco_env/ik.py` | 逆运动学求解器（阻尼最小二乘法） |
| `model/demo_scene.xml` | MuJoCo 场景定义（机器人 + 桌面 + 物体） |
| `model/eco65_with_pgc140_d435i.xml` | ECO65 机械臂 + PGC140 夹爪 + D435i 相机模型 |
| `ckpt/act_y/config.json` | ACT 策略配置参数 |
