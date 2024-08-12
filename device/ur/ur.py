import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import numpy as np


class UR:
    def __init__(self, ip: str) -> None:
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)

    def applyTcpVel(self, vel, acc=0.25, time=0.0):
        base_vel = self._tcpVelToBaseVel(vel)
        self.rtde_c.speedL(base_vel, acc, time)

    def _tcpVeltoBaseVel(self, tcp_vel):
        tcp_pose_vec = self.rtde_r.getActualTCPPose()
        rot_matrix = R.from_rotvec(tcp_pose_vec[3:]).as_matrix()
        trans_matrix = np.eye(6)
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[3:, 3:] = rot_matrix
        return trans_matrix @ tcp_vel

    def applyCameraVel(self, vel, extrinsic_matrix, acc=0.25, time=0.0):
        linear_vel = vel[:3]
        angular_vel = vel[3:]
        angular_tcp_vel = extrinsic_matrix[:3, :3] * angular_vel
        linear_tcp_vel = extrinsic_matrix[:3, :3] * linear_vel - np.cross(
            extrinsic_matrix[:3, 3], angular_tcp_vel
        )
        tcp_vel = np.concatenate((linear_tcp_vel, angular_tcp_vel))
        self.applyTcpVel(tcp_vel, acc, time)

    def stop(self):
        self.rtde_c.stopL()
        self.rtde_c.stopJ()

    def close(self):
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
