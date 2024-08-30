import numpy as np
import rtde_control
import rtde_receive
from ...algo.transforms import *
from scipy.spatial.transform import Rotation as R
from .manipulator_base import Manipulator


class UR(Manipulator):
    def __init__(self, ip: str, base_to_world: np.ndarray = np.eye(4)) -> None:
        self.base_to_world = base_to_world
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)

    @property
    def world_pose(self) -> np.ndarray:
        pose_vec = self.rtde_r.getActualTCPPose()
        trans_matrix = np.eye(4)
        rot_matrix = R.from_rotvec(pose_vec[3:]).as_matrix()
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3] = pose_vec[:3]
        return self.base_to_world @ trans_matrix

    def applyTcpVel(self, tcp_vel: np.ndarray, acc=0.25, time=0.0) -> None:
        pose_vec = self.rtde_r.getActualTCPPose()
        rot_matrix = R.from_rotvec(pose_vec[3:]).as_matrix()
        base_vel = velTransform(tcp_vel, rot_matrix)
        self.rtde_c.speedL(base_vel, acc, time)

    def applyVel(self, vel: np.ndarray, acc=0.25, time=0.0) -> None:
        self.rtde_c.speedL(vel, acc, time)

    def applyWorldVel(self, world_vel: np.ndarray, acc=0.25, time=0.0) -> None:
        pose_vec = self.rtde_r.getActualTCPPose()
        rot_matrix = self.base_to_world @ R.from_rotvec(pose_vec[3:]).as_matrix()
        world_vel = velTransform(world_vel, rot_matrix)
        self.rtde_c.speedL(world_vel, acc, time)

    @property
    def tcp_pose(self) -> np.ndarray:
        pose_vec = self.rtde_r.getActualTCPPose()
        trans_matrix = np.eye(4)
        rot_matrix = R.from_rotvec(pose_vec[3:]).as_matrix()
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3] = pose_vec[:3]
        return trans_matrix

    def stop(self, acc=10.0) -> None:
        self.rtde_c.stopL(acc)
        self.rtde_c.stopJ(acc)

    def disconnect(self) -> None:
        self.stop()
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
