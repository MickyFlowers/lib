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
    def pose(self) -> np.ndarray:
        pass
