import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SimpleEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle'):
        """
        参数:
            xml_path: str, XML 模型文件路径
            action_type: str, 动作空间类型 ('eef_pose'/'delta_joint_angle'/'joint_angle')
            state_type: str, 状态空间类型 ('joint_angle'/'ee_pose')
        """
        # 切换到 XML 所在目录加载，确保 MuJoCo include/mesh 路径正确解析
        xml_abs_path = os.path.abspath(xml_path)
        xml_dir = os.path.dirname(xml_abs_path)
        xml_name = os.path.basename(xml_abs_path)
        cwd = os.getcwd()
        try:
            os.chdir(xml_dir)
            self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_name)
        finally:
            os.chdir(cwd)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names = ['joint_1',
                    'joint_2',
                    'joint_3',
                    'joint_4',
                    'joint_5',
                    'joint_6',]
        self.init_viewer()
        self.reset()

    def init_viewer(self):
        '''
        初始化渲染窗口
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self):
        '''
        重置环境：将机器人移至初始位置，并随机放置物体
        '''
        q_init = np.deg2rad([0,0,0,0,0,0])
        # ECO65 预备位姿：夹爪指向桌面
        q_zero = np.deg2rad([0.0, 0.0, 90.0, -0.61, -90.01, 90.0])
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

        # 随机放置物体
        obj_names = self.env.get_body_names(prefix='body_obj_')
        n_obj = len(obj_names)
        obj_xyzs = sample_xyzs(
            n_obj,
            x_range   = [+0.20,+0.35],
            y_range   = [-0.15,+0.15],
            z_range   = [0.83,0.83],
            min_dist  = 0.18,
            xy_margin = 0.0
        )
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
            self.env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))
        self.env.forward(increase_tick=False)

        # 设置机器人初始位姿
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([0.0]*1)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='link_6')
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        print("-" * 50)
        print("[遥操作按键绑定]")
        print("  平移 (X/Y 平面):     W/S = 前进/后退,  A/D = 左移/右移")
        print("  平移 (Z 轴):         R = 上升,  F = 下降")
        print("  旋转 (末端):         Q = 左倾,  E = 右倾")
        print("                        ↑/↓ = 俯仰,  ←/→ = 偏航")
        print("  夹爪:               SPACE = 开合")
        print("  重置:               Z = 重置当前 episode")
        print("-" * 50)
        self.gripper_state = False
        self.past_chars = []

    def step(self, action):
        '''
        在环境中执行一步动作
        参数:
            action: np.array, 形状 (7,), 要执行的动作
        返回:
            state: np.array, 执行动作后的环境状态
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]
        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'link_6',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')

        gripper_val = action[-1] * 255.0
        gripper_cmd = np.array([gripper_val], dtype=np.float32)
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        从环境中抓取图像
        返回:
            rgb_agent: np.array, 第三人称视角 RGB 图像
            rgb_ego: np.array, 腕部相机 RGB 图像
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='d435i_rgb')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False):
        '''
        渲染环境画面
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='link_6')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
            joint_angles = self.env.get_qpos_joints(joint_names=self.joint_names)
            joint_angles_deg = np.rad2deg(joint_angles)
            angle_str = ' '.join([f'{j:.0f}' for j in joint_angles_deg])
            self.env.viewer_text_overlay(text1='Joint Angles', text2=angle_str)
        self.env.render()

    def get_joint_state(self):
        '''
        获取机器人关节状态
        返回:
            q: np.array, 关节角 + 夹爪状态 (0=闭合, 1=张开)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('right_driver_joint')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        通过键盘遥操作控制机器人
        返回:
            action: np.array, 要执行的动作
            done: bool, 是否重置遥操作

        按键:
            ---------     -----------------------
               w       ->        前进
            s  a  d        左移   后退   右移
            ---------      -----------------------
            X/Y 平面平移

            ---------
            R: 上升
            F: 下降
            ---------
            Z 轴平移

            ---------
            Q: 左倾
            E: 右倾
            ↑: 上仰
            ↓: 下俯
            →: 右转
            ←: 左转
            ---------
            末端旋转

            ---------
            Z: 重置
            SPACE: 夹爪开合
            ---------


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        获取机器人关节角增量
        返回:
            delta: np.array, 关节角增量 + 夹爪状态 (0=闭合, 1=张开)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('right_driver_joint')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    # 判断任务是否成功
    def check_success(self):
        '''
        ['body_obj_mug_5', 'body_obj_plate_11']
        判断杯子是否已放置在盘子上，且夹爪已张开、机械臂已抬起
        '''
        # 获取物体当前位置
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')

        if np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6 and self.env.get_qpos_joint('right_driver_joint') < 0.1:
            p = self.env.get_p_body('link_6')[2]
            if p > 0.9:
                return True
        return False
    
    def get_obj_pose(self):
        '''
        返回:
            p_mug: np.array, 杯子位置
            p_plate: np.array, 盘子位置
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return p_mug, p_plate

    # 设置杯子和盘子的位置和方向
    def set_obj_pose(self, p_mug, p_plate):
        '''
        设置物体位姿
        参数:
            p_mug: np.array, 杯子位置
            p_plate: np.array, 盘子位置
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()

    # 获取末端执行器的位姿
    def get_ee_pose(self):
        '''
        获取机器人末端执行器位姿
        '''
        p, R = self.env.get_pR_body(body_name='link_6')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
