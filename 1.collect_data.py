import sys
import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_NAME = 'eco65_pnp'
NUM_DEMO = 5 # 采集演示条数
ROOT = "./demo_data" # 演示数据保存根目录

TASK_NAME = 'Put mug cup on the plate'
xml_path = './model/demo_scene.xml'
# 初始化仿真环境
PnPEnv = SimpleEnv(xml_path, state_type = 'joint_angle')

create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False


if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT,
                robot_type="eco65_2f85",
                fps=20, # 每秒 20 帧
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"], # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # 6 个关节角 + 1 个夹爪
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["obj_init"], # 物体初始位置，训练时不使用
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

action = np.zeros(7)
episode_id = 0
record_flag = False # 机器人开始移动后再录制
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # 检查 episode 是否结束
        done = PnPEnv.check_success()
        if done:
            # 保存 episode 数据并重置环境
            dataset.save_episode()
            PnPEnv.reset()
            episode_id += 1
        # 遥操作机器人，获取末端增量位姿和夹爪状态
        action, reset  = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # 重置环境并清空 episode 缓冲区
            # 可按 'z' 键触发
            PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # 推进仿真，获取末端位姿和图像
        ee_pose = PnPEnv.get_ee_pose()
        agent_image,wrist_image = PnPEnv.grab_image()
        # 缩放至 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        if record_flag:
            # 将当前帧写入数据集
            dataset.add_frame( {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose,
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    "task": TASK_NAME,
                },
            )
        PnPEnv.render(teleop=True)

PnPEnv.env.close_viewer()
dataset.finalize()
# 清理临时图像文件夹
import shutil
images_dir = dataset.root / 'images'
if images_dir.exists():
    shutil.rmtree(images_dir)
