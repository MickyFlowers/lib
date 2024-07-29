import os
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


class SAM:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        model_type: str = "default",
        model_path: str = "sam_vit_h_4b8939.pth",
        device: str = "cuda:0",
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.output_files = os.listdir(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

        if os.path.isdir(self.input_path):
            self.img_paths = glob.glob(os.path.join(self.input_path, "*.jpg"))
            import re

            def extract_number(filepath):
                match = re.search(r"img-(\d+).jpg", filepath)
                if match:
                    return int(match.group(1))
                return 0

            self.img_paths.sort(key=extract_number)
            self.img_paths = [
                self.img_paths[i]
                for i in range(0, len(self.img_paths))
                if os.path.basename(self.img_paths[i]) not in self.output_files
            ]

        else:
            self.img_paths = [self.input_path]
        assert model_type in ["default", "vit_h", "vit_b", "vit_l"]
        print(f"Loading Model...")
        print(f"Model Type: {model_type}")
        print(f"Model Checkpoint: {model_path}")
        model_path = os.path.join(os.path.dirname(__file__), "model", model_path)
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.points = []
        self.background_points = []
        self._current_img_idx = 0

    def _key_press_event(self, event):
        if event.key == "c":
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            mask_img = np.zeros_like(self.img)
            if self.mask is not None:
                mask_img[self.mask == 1] = self.img[self.mask == 1]
            cv2.imwrite(
                os.path.join(
                    self.output_path,
                    os.path.basename(self.img_paths[self._current_img_idx]),
                ),
                mask_img,
            )
            self.points = []
            self.background_points = []
            self._current_img_idx += 1
            self.img = cv2.imread(self.img_paths[self._current_img_idx])
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(self.img)
            plt.cla()
            plt.imshow(self.img)
            self.fig.canvas.draw()

    def _mouse_click_event(self, event):
        mask = None
        if event.inaxes is None:
            return
        if event.button == plt.MouseButton.LEFT:
            if event.xdata is not None and event.ydata is not None:
                self.points.append([event.xdata, event.ydata])
                mask, _, _ = self._predict(self.img)
                mask = mask[0].astype(np.uint8)
        elif event.button == plt.MouseButton.MIDDLE:
            if event.xdata is not None and event.ydata is not None:
                self.background_points.append([event.xdata, event.ydata])
                mask, _, _ = self._predict(self.img)
                mask = mask[0].astype(np.uint8)
        elif event.button == plt.MouseButton.RIGHT:
            if event.xdata is not None and event.ydata is not None:
                min_distance = np.inf
                background_min_distance = np.inf
                if self.points:
                    distances = [
                        np.sqrt((p[0] - event.xdata) ** 2 + (p[1] - event.ydata) ** 2)
                        for p in self.points
                    ]
                    min_distance = np.min(distances)
                if self.background_points:
                    background_distances = [
                        np.sqrt((p[0] - event.xdata) ** 2 + (p[1] - event.ydata) ** 2)
                        for p in self.background_points
                    ]
                    background_min_distance = np.min(background_distances)
                if background_min_distance < min_distance:
                    if background_min_distance < 30:
                        self.background_points.pop(np.argmin(background_distances))
                else:
                    if min_distance < 30:
                        self.points.pop(np.argmin(distances))
                if self.points + self.background_points:
                    mask, _, _ = self._predict(self.img)
                    mask = mask[0].astype(np.uint8)
        plt.cla()
        plt.imshow(self.img)
        if mask is not None:
            plt.imshow(mask, cmap="cividis", alpha=0.5)
        self.mask = mask
        if self.points:
            x, y = zip(*self.points)
            plt.scatter(x, y, c="r", s=20)
        if self.background_points:
            x, y = zip(*self.background_points)
            plt.scatter(x, y, c="b", s=20)
        self.fig.canvas.draw()

    def _predict(self, img: np.ndarray) -> np.ndarray:
        result = self.predictor.predict(
            point_coords=np.array(self.points + self.background_points),
            point_labels=np.array(
                [1] * len(self.points) + [0] * len(self.background_points)
            ),
            multimask_output=False,
        )
        return result

    def segment(self):
        window_name = "segment anything"
        self.fig = plt.figure(window_name)
        self.fig.canvas.mpl_connect("button_press_event", self._mouse_click_event)
        self.fig.canvas.mpl_connect("key_press_event", self._key_press_event)

        self.img = cv2.imread(self.img_paths[self._current_img_idx])
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.img)
        plt.imshow(self.img)
        while True:
            plt.pause(0.001)
