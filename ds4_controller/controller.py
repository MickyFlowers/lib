from enum import Enum
import inspect


class ButtonEvent(Enum):
    PRESS = 0
    RELEASE = 1
    PRESS_HOLD = 2
    RELEASE_HOLD = 3


class Button:
    def __init__(self) -> None:
        super().__init__()
        self._button = 0
        self._last_button_hold = 0
        self.event = ButtonEvent.RELEASE_HOLD

    def update(self, button):
        self._last_button_hold = self._button
        self._button = button
        if self._button == 0 and self._last_button_hold == 1:
            self.event = ButtonEvent.RELEASE
        elif self._button == 0 and self._last_button_hold == 0:
            self.event = ButtonEvent.RELEASE_HOLD
        elif self._button == 1 and self._last_button_hold == 0:
            self.event = ButtonEvent.PRESS
        elif self._button == 1 and self._last_button_hold == 1:
            self.event = ButtonEvent.PRESS_HOLD


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
        self.buttons = {name: Button() for name in self._button_names}

    def update(self, data) -> None:
        for name, obj in inspect.getmembers(data):
            if name in self._button_names:
                self.buttons[name].update(obj)
