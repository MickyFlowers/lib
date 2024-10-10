from abc import ABC, abstractmethod
import numpy as np


class Manipulator(ABC):
    @abstractmethod
    def world_pose(self) -> np.ndarray:
        pass

    @abstractmethod
    def applyTcpVel(self, tcp_vel, acc, time):
        pass

    @abstractmethod
    def applyVel(self, vel, acc, time):
        pass

    @abstractmethod
    def applyWorldVel(self, world_vel, acc, time):
        pass

    @abstractmethod
    def tcp_pose(self) -> np.ndarray:
        pass

    @abstractmethod
    def moveToWorldPose(self, pose, vel, acc, asynchronous):
        pass

    @abstractmethod
    def moveToPose(self, pose, vel, acc, asynchronous):
        pass
