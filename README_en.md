# ECO65 Imitation Learning: ACT & pi0 Dual-Policy Pick-and-Place

<p align="center">
  <img src="docs/1.png" width="640" alt="ECO65 MuJoCo Simulation Screenshot">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/MuJoCo-3.2+-green?logo=robot-framework" alt="MuJoCo">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/LeRobot-v3.0-orange?logo=huggingface" alt="LeRobot">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

<p align="center">
  <a href="README.md">简体中文</a> | <b>English</b>
</p>

---

### Overview

This project implements a complete imitation learning pipeline: controlling an ECO65 6-DOF robotic arm in the MuJoCo simulation environment via keyboard teleoperation to perform the **"Put mug cup on the plate"** pick-and-place task, collecting demonstration data, training an ACT (Action Chunking with Transformers) policy network, and finally deploying the model for autonomous task completion.

The core workflow: **Data Collection → Data Visualization → Policy Training (ACT / pi0) → Model Deployment**, built on the LeRobot dataset framework (v3.0 format) and MuJoCo physics engine.

### Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Observation Views](#observation-views)
- [Features](#features)
- [Dependencies](#dependencies)
- [Environment Setup](#environment-setup)
- [Installation & Build](#installation--build)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [License](#license)

---

### Architecture

```
┌─────────────────────────────────────────────────────┐
│              Data Collection (1.collect_data.py)     │
│  Teleop → IK Solver → MuJoCo Sim → LeRobot Dataset  │
│  Input: Keyboard (WASD + QE + Arrows + Space)       │
│  Output: demo_data/ (Parquet + MP4)                 │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│           Data Visualization (2.visualize_data.py)   │
│  Replay episodes, verify data quality               │
│  Compute normalization stats → stats.json            │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│          Training (3_act_train.py / 5_pi0_train.py)  │
│  ACT: ResNet18 + Transformer Encoder-Decoder         │
│  pi0: VLA foundation model (pretrained fine-tuning)  │
│  Input: RGB image (256×256) + end-effector pose (6D) │
│  Output: joint angles + gripper (7D) × action chunk  │
│  Save: ckpt/act_y/ or ckpt/pi0_eco65/                │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│            ACT Deployment (4.deploy.py)               │
│  Load ACT model → Inference → MuJoCo execution → Task │
│  + Temporal Ensemble smoothing + Success detection   │
└─────────────────────────────────────────────────────┘
```

**ACT Network Architecture:**

```
Input                        Output
┌──────────────┐             ┌──────────────────┐
│ image (3×256)│──┐          │  action chunk    │
│   ResNet18   │  │   ┌────┐ │  [10, 7]         │
│   ─────────  │──┼──▶│    │ │  t₀..t₉: joints  │
│  img_feat(512)│  │  │ACT │ │  + gripper       │
└──────────────┘  │  │    │ └──────────────────┘
                  │  │Tran│
┌──────────────┐  │  │sfo─│
│ state [6]    │──┤  │rmer│
│  (x,y,z,r,p,y)│  │  │Enco│
└──────────────┘  │  │der─│
                  │  │Deco │
         ┌──────┐ │  │der │
         │ VAE  │ │  │    │
         │ (32D) │─┘  │    │
         └──────┘    └────┘
```

### Observation Views

<p align="center">
  <img src="docs/agent_view.png" width="400" alt="Agent View">
  <img src="docs/wrist_view.png" width="400" alt="Wrist View">
</p>
<p align="center">
  <b>Agent View</b> (third-person)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>Wrist View</b> (wrist-mounted camera)
</p>

| View | Camera Source | Purpose |
|------|--------------|---------|
| Agent View | MuJoCo `agentview` | Global scene awareness — object positions and robot arm state |
| Wrist View | MuJoCo `d435i_rgb` | Close-up end-effector view — precise gripper-object spatial relationship |

### Features

- **End-to-end imitation learning pipeline**: Four scripts cover data collection through model deployment
- **Keyboard teleoperation**: Intuitive WASD + QE end-effector control — no specialized hardware required
- **Dual-policy support**: ACT (Action Chunking with Transformers) and pi0 (VLA foundation model) for different data scales
- **ACT policy**: Transformer-based Action Chunking with 10-step prediction and VAE encoding for smooth trajectories
- **pi0 pretrained fine-tuning**: Large-scale robot data pretrained, only 10-20 demos needed to generalize to new tasks
- **Temporal Ensemble**: Exponential moving average smoothing during deployment eliminates prediction jitter
- **LeRobot v3.0 format**: Standardized Parquet + MP4 storage, compatible with HuggingFace LeRobot ecosystem
- **High-fidelity MuJoCo simulation**: ECO65 6-axis arm + Robotiq 2F-85 gripper + D435i depth camera
- **IK solver**: Damped Least Squares (DLS) inverse kinematics for end-effector space teleoperation
- **Dual-view rendering**: Third-person view (agentview) + wrist view (d435i_rgb)
- **Randomized object positions**: Mug and plate positions are randomized within table bounds on each reset

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | ≥ 3.10 | Main programming language |
| PyTorch | ≥ 2.0 | Deep learning framework |
| MuJoCo | ≥ 3.2 | Physics simulation engine |
| LeRobot | ≥ 0.4 | Dataset management & policy framework |
| NumPy | ≥ 1.24 | Numerical computing |
| OpenCV (cv2) | ≥ 4.8 | Image processing |
| Pillow | ≥ 10.0 | Image I/O |
| GLFW | - | MuJoCo window rendering backend |
| Matplotlib | ≥ 3.7 | Training curves & data visualization |
| TorchVision | ≥ 0.15 | ResNet pretrained weights |

### Environment Setup

We recommend using Conda:

```bash
# Create and activate environment
conda create -n eco65_act python=3.10 -y
conda activate eco65_act

# Install PyTorch (choose based on CUDA version)
# CUDA 12.1
pip install torch torchvision --index-url https://pytorch.org/whl/cu121
# Or CPU only
pip install torch torchvision --index-url https://pytorch.org/whl/cpu

# Install MuJoCo
pip install mujoco

# Install LeRobot
pip install lerobot

# Install other dependencies
pip install numpy opencv-python pillow glfw matplotlib
```

### Installation & Build

```bash
# Clone the repository
git clone https://github.com/yakousansan/eco65-pnp.git
cd eco65-pnp

pip install -e .
```

### Usage

#### Step 1: Collect Demonstration Data

```bash
python 1.collect_data.py
```

In the MuJoCo window, use keyboard teleoperation to control the robot arm for the "put mug on plate" task:

| Key | Action | Description |
|-----|--------|-------------|
| `W` / `S` | Forward / Backward | World X axis, step 0.007m |
| `A` / `D` | Left / Right | World Y axis, step 0.007m |
| `R` / `F` | Up / Down | World Z axis, step 0.007m |
| `Q` / `E` | Tilt Left / Right | Rotate around Z axis (roll) |
| `↑` / `↓` | Pitch | Rotate around X axis |
| `←` / `→` | Yaw | Rotate around Y axis |
| `Space` | Gripper Open/Close | Toggle 0 ↔ 1 |
| `Z` | Reset | Discard current episode, restart |
| `Esc` | Quit | Close window, auto-save data |

Each successful placement auto-saves one episode. Closing the window triggers video encoding and disk write.

<p align="center">
  <img src="docs/episode_03_agent.gif" width="380" alt="Agent View">
  <img src="docs/episode_03_wrist.gif" width="380" alt="Wrist View">
</p>

> **Tip**: Modify `NUM_DEMO` to change collection count, `ROOT` to change save path in the script.

#### Step 2: Visualize Collected Data

```bash
python 2.visualize_data.py
```

Replay collected episodes in MuJoCo to verify data quality. Automatically computes normalization statistics (mean/std) and saves to `demo_data/meta/stats.json`.

<p align="center">
  <img src="docs/episode_01_agent.gif" width="480" alt="Visualization replay demo">
</p>

#### Step 3: Train Policy (ACT or pi0)

**ACT training:**

```bash
python 3_act_train.py
```

**pi0 training:**

```bash
python 5_pi0_train.py
```

ACT training:
- Load dataset and statistics
- Initialize ACT network (ResNet18 + Transformer + VAE)
- 3000 training steps, Adam optimizer (lr=1e-4), batch size=64
- Data augmentation: Gaussian image noise (std=0.02)
- Loss: L1 reconstruction + KL divergence (weight 10.0)
- GPU training takes ~5-15 minutes
- Model saved to `ckpt/act_y/`

pi0 training:
- Auto-loads `lerobot/pi0` pretrained weights (large-scale robot data)
- 20000 fine-tuning steps, batch size=16 (adjustable by VRAM)
- Only 10-20 demos needed for generalization
- Model saved to `ckpt/pi0_eco65/`

After training, a prediction vs. ground truth comparison plot is displayed.

#### Step 4: Deploy and Test (ACT)

```bash
python 4.deploy.py
```

Loads the trained **ACT** policy (`ckpt/act_y/`) and runs autonomously in MuJoCo. The policy predicts action chunks from real-time visual and state inputs. Automatically resets on task completion.

<p align="center">
  <img src="docs/deploy_demo.gif" width="480" alt="Deployment demo">
</p>

> Demo above based on 10 teleoperated demonstration trajectories.

### Project Structure

```
eco65-pnp/
├── 1.collect_data.py         # Data collection (keyboard teleop + LeRobot recording)
├── 2.visualize_data.py       # Data visualization (replay + stats computation)
├── 3_act_train.py            # ACT policy training
├── 4.deploy.py               # ACT policy deployment & inference
├── 5_pi0_train.py            # pi0 policy training
├── pi0_eco65.yaml            # pi0 training config
├── model/                    # MuJoCo model assets
│   ├── demo_scene.xml        # Main scene (table + robot + objects)
│   ├── eco65_with_2f85_d435i.xml  # ECO65 + Robotiq 2F-85 + D435i model
│   ├── mug_5/                # Mug mesh model
│   ├── plate_11/             # Plate mesh model
│   ├── tabletop/             # Table mesh model
│   ├── realsense_d435i/      # D435i camera model
│   ├── robotiq_2f85/         # Robotiq 2F-85 gripper model
│   └── eco65_meshes/         # ECO65 arm meshes
├── mujoco_env/               # MuJoCo environment wrapper
│   ├── __init__.py
│   ├── y_env.py              # SimpleEnv (teleop, IK, rendering)
│   ├── mujoco_parser.py      # Low-level MuJoCo interface
│   ├── ik.py                 # IK solver (Damped Least Squares)
│   ├── transforms.py         # Coordinate transforms (RPY ↔ rotation matrix)
│   └── utils.py              # Utilities (sampling, imaging, rendering)
├── ckpt/
│   └── act_y/                # ACT policy checkpoint
│       └── config.json       # Policy configuration (model weights generated via training)
└── demo_data/                # LeRobot dataset (generated after collection)
    └── meta/
        └── info.json         # Dataset metadata
```

### Configuration

#### Data Collection Parameters (`1.collect_data.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_DEMO` | `1` | Number of episodes to collect |
| `ROOT` | `"./demo_data"` | Dataset save path |
| `TASK_NAME` | `"Put mug cup on the plate"` | Task description |
| Collection FPS | 20 Hz | Frames per second |
| Image Resolution | 256 × 256 | Captured image size |

#### ACT Policy Parameters (`3_act_train.py` and `ckpt/act_y/config.json`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunk_size` | 10 | Predict 10 action steps at once |
| `n_action_steps` | 10 (train) / 1 (deploy) | Action steps to execute |
| `n_obs_steps` | 1 | Observation history steps |
| `dim_model` | 512 | Transformer hidden dimension |
| `n_heads` | 8 | Multi-head attention heads |
| `n_encoder_layers` | 4 | Encoder layers |
| `n_decoder_layers` | 1 | Decoder layers |
| `dim_feedforward` | 3200 | FFN intermediate dimension |
| `dropout` | 0.1 | Dropout rate |
| `kl_weight` | 10.0 | VAE KL divergence weight |
| `latent_dim` | 32 | VAE latent space dimension |
| `vision_backbone` | resnet18 | Vision backbone |
| `learning_rate` | 1e-4 | Adam learning rate |
| `training_steps` | 3000 | Training iterations |
| `batch_size` | 64 | Batch size |
| `temporal_ensemble_coeff` | 0.9 (deploy only) | Temporal ensemble smoothing |

#### Teleoperation Parameters (`mujoco_env/y_env.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Move step | 0.007 m | End-effector translation per keypress |
| Rotate step | ~1.7° | End-effector rotation per keypress |
| IK max iterations | 50 | Max IK solver iterations |
| IK tolerance | 1e-2 m / 5° | Position/orientation convergence threshold |

### License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.
