import cv2
import math
import time
import threading

import json
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

from perception import Camera, CameraIntrinsic, Frame
from ur5 import UR5


def load_scenes():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF('plane.urdf', [0, 0, -0.63], [0, 0, 0, 1])
    table0 = p.loadURDF('table/table.urdf', [0, 0.5, -0.63], [0, 0, 0, 1])
    table1 = p.loadURDF('table/table.urdf', [0, -0.5, -0.63], [0, 0, 0, 1])
    bucket = p.loadURDF('tray/tray.urdf', [-0.4, 0.5, 0], [0, 0, 0, 1])

    def rand_distribute(file_name, x_min=-0.2, x_max=0.1, y_min=-0.1, y_max=0.1, z_min=0.2, z_max=0.8, scale=1.0):
        xyz = np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max], size=3)
        rpy = np.random.uniform(-np.pi, np.pi, size=3)
        orn = p.getQuaternionFromEuler(rpy)
        object_id = p.loadURDF(file_name, xyz, orn, globalScaling=scale)

        return object_id

    objects_id = []
    for i in range(3):
        objects_id.append(rand_distribute('lego/lego.urdf'))

    for i in range(2):
        objects_id.append(rand_distribute('jenga/jenga.urdf'))

    for i in range(2):
        objects_id.append(rand_distribute('cube_small.urdf'))

    return objects_id


def setup_target_pose_params(initial_xyz, initial_rpy):
    initial_x, initial_y, initial_z = initial_xyz
    initial_roll, initial_pitch, initial_yaw = initial_rpy

    param_ids = [
        p.addUserDebugParameter('x', -1, 1, initial_x),
        p.addUserDebugParameter('y', -1, 1, initial_y),
        p.addUserDebugParameter('z', 0, 1, initial_z),
        p.addUserDebugParameter('roll', -math.pi, math.pi, initial_roll),
        p.addUserDebugParameter('pitch', -math.pi, math.pi, initial_pitch),
        p.addUserDebugParameter('yaw', -math.pi, math.pi, initial_yaw),
        p.addUserDebugParameter('finger openness', 0, 1, 1)
    ]

    return param_ids


def read_user_params(param_ids):
    return [p.readUserDebugParameter(param_id) for param_id in param_ids]


class DebugAxes(object):
    """
    可视化某个局部坐标系, 红色x轴, 绿色y轴, 蓝色z轴
    """
    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Arguments:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos)
        rot3x3 = R.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])


class Environment(object):
    def __init__(self):
        self.urdf_file = "./urdf/real_arm.urdf"
        self.camera_config = "./setup.json"

        with open(self.camera_config, "r") as j:
            config = json.load(j)
        camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])

        self.client = p.connect(p.GUI)
        p.setRealTimeSimulation(1)
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(1.674, 70, -50.8, [0, 0, 0])

        self.objects = load_scenes()
        self.camera = Camera(camera_intrinsic)
        self.arm = UR5(self.urdf_file)

        self.end_axes = DebugAxes()  # 机械臂末端的局部坐标系
        self.camera_axes = DebugAxes()  # 相机坐标系

        # thread for updatig debug axes
        self.update_debug_axes_thread = threading.Thread(
            target=self.update_debug_axes)
        self.update_debug_axes_thread.setDaemon(True)
        self.update_debug_axes_thread.start()

        # thread for updating camera image
        self.update_camera_image_thread = threading.Thread(
            target=self.update_camera_image)
        self.update_camera_image_thread.setDaemon(True)
        self.update_camera_image_thread.start()

    def _bind_camera_to_end(self, end_pos, end_orn):
        """设置相机坐标系与末端坐标系的相对位置
        
        Arguments:
        - end_pos: len=3, end effector position
        - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

        Returns:
        - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
        """
        relative_offset = [-0.05, 0, 0.1]  # 相机原点相对于末端执行器局部坐标系的偏移量
        end_orn = R.from_quat(end_orn).as_matrix()
        end_x_axis, end_y_axis, end_z_axis = end_orn.T

        wcT = np.eye(4)  # w: world, c: camera, ^w_c T
        wcT[:3, 0] = -end_y_axis  # camera x axis
        wcT[:3, 1] = -end_z_axis  # camera y axis
        wcT[:3, 2] = end_x_axis  # camera z axis
        wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position
        return wcT

    def update_debug_axes(self):
        while True:
            # update debug axes and camera position
            end_pos, end_orn = self.arm.get_end_state()
            self.end_axes.update(end_pos, end_orn)

            wcT = self._bind_camera_to_end(end_pos, end_orn)
            self.camera_axes.update(
                pos=wcT[:3, 3],
                orn=R.from_matrix(wcT[:3, :3]).as_quat()
            )
    
    def update_camera_image(self):
        cv2.namedWindow("image")
        while True:
            end_pos, end_orn = self.arm.get_end_state()
            wcT = self._bind_camera_to_end(end_pos, end_orn)
            cwT = np.linalg.inv(wcT)

            frame = self.camera.render(cwT)
            assert isinstance(frame, Frame)

            rgb = frame.color_image()  # 这里以显示rgb图像为例, frame还包含了深度图, 也可以转化为点云
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # flip the rgb channel
            cv2.imshow("image", bgr)
            key = cv2.waitKey(1)
            time.sleep(0.02)
            
    def start_manual_control(self):
        init_xyz = [0.08, -0.20, 0.6]
        init_rpy = [0, math.pi / 2., 0]
        param_ids = setup_target_pose_params(init_xyz, init_rpy)

        while True:
            target_pose = read_user_params(param_ids)  # [x, y, z, roll, pitch, yaw, finger openness]
            self.arm.move_to(target_pose[:3],
                            p.getQuaternionFromEuler(target_pose[3:6]))
            self.arm.control_gripper(target_pose[-1])
            time.sleep(0.02)


if __name__ == "__main__":
    env = Environment()
    time.sleep(2)

    print("[INFO] Start manual control!")
    env.start_manual_control()

