import inspect
from ..device_base.button import Button


class Ds4Controller:
    def __init__(self) -> None:
        self._button_names = [
            "button_dpad_up",
            "button_dpad_down",
            "button_dpad_left",
            "button_dpad_right",
            "button_cross",
            "button_circle",
            "button_triangle",
            "button_square",
            "button_l1",
            "button_l2",
            "button_l3",
            "button_r1",
            "button_r2",
            "button_r3",
            "button_share",
            "button_options",
            "button_trackpad",
            "button_ps",
        ]
        self._axis_names = [
            "axis_left_x",
            "axis_left_y",
            "axis_right_x",
            "axis_right_y",
        ]
        self._plug_names = [
            "plug_usb",
            "plug_audio",
            "plug_mic",
        ]
        self.buttons = {name: Button() for name in self._button_names}
        self.plugs = {name: 0 for name in self._plug_names}
        self.axis = {name: 0.0 for name in self._axis_names}
        self.battery_percentage = 0.0

    def update(self, data) -> None:
        for name, obj in inspect.getmembers(data):
            if name in self._button_names:
                self.buttons[name].update(obj)
            elif name in self._plug_names:
                self.plugs[name].update({name: obj})
            elif name in self._axis_names:
                self.axis[name].update({name: obj})
            elif name == "battery_percentage":
                self.battery_percentage = obj
