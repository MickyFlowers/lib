from abc import ABC, abstractmethod


class VisualServoControllerBase(ABC):

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def calc_vel(self):
        pass



