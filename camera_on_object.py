import cv2
import json
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from perception import Camera, CameraIntrinsic, Frame


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
        rot3x3 = Rotation.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])


class BindCamera(Camera):
    def __init__(self, obj_id, intrinsic, near=0.01, far=4, rela_tform=None):
        """
        Arguments:
        - obj_id: int, object uid generate from p.loadURDF()
        - intrinsic: CameraIntrinsic object
        - rela_tform: 4x4 relative transform matrix to binding object
        """
        super(BindCamera, self).__init__(intrinsic, near, far)
        self.obj_id = obj_id
        self.rela_tform = rela_tform
    
    def object_pose(self):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj_id)

        woT = np.eye(4)
        woT[:3, :3] = Rotation.from_quat(obj_orn).as_matrix()
        woT[:3, 3] = np.array(obj_pos)

        return woT
    
    def extrinsic(self):
        woT = self.object_pose()
        ocT = self.rela_tform if self.rela_tform is not None else np.eye(4)
        wcT = np.dot(woT, ocT)

        return wcT
    
    def render(self):
        wcT = self.extrinsic()
        cwT = np.linalg.inv(wcT)

        return super(BindCamera, self).render(cwT)
    

if __name__ == "__main__":
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(1.674, 70, -50.8, [0, 0, 0])

    plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])  # 地面
    cube = p.loadURDF("cube_small.urdf", [0, 0, 0.2], [0, 0, 0, 1])  # 小方块，待会儿相机就绑在这上面

    camera_config = "./setup.json"
    with open(camera_config, "r") as j:
        config = json.load(j)
    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])

    #### 手动指定相机相对于绑定物体的相对位姿
    rela_tform = np.eye(4)
    rela_tform[:3, 0] = [1, 0, 0]     # 相机x轴与物体x轴同向
    rela_tform[:3, 1] = [0, 0, -1]    # 相机y轴与物体z轴反向
    rela_tform[:3, 2] = [0, 1, 0]     # 相机z轴与物体y轴同向
    rela_tform[:3, 3] = [0, 0, 0.08]  # 相机原点在立方体上方8cm
    ####

    camera = BindCamera(cube, camera_intrinsic, near=0.1, far=15, rela_tform=rela_tform)

    debug_object_axes = DebugAxes()
    debug_camera_axes = DebugAxes()

    while True:
        object_pose = camera.object_pose()
        debug_object_axes.update(object_pose[:3, 3], Rotation.from_matrix(object_pose[:3, :3]).as_quat())
        
        extrinsic = camera.extrinsic()
        debug_camera_axes.update(extrinsic[:3, 3], Rotation.from_matrix(extrinsic[:3, :3]).as_quat())

        frame = camera.render()
        assert isinstance(frame, Frame)

        rgb = frame.color_image()  # 这里以显示rgb图像为例, frame还包含了深度图, 也可以转化为点云
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # flip the rgb channel
        cv2.imshow("image", bgr)
        key = cv2.waitKey(1)

        # 让相机绑定的方块边移动边旋转
        p.resetBaseVelocity(cube, [0.4, 0.4, 0], [0, 0, 0.5])
