from enum import Enum


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
