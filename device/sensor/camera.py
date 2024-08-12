import numpy as np
import pyrealsense2 as rs
import cv2
import os


class Camera:
    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.color_img = None
        self.depth_img = None

    def set_img(self, color_img, depth_img):
        self.color_img = np.asanyarray(color_img)
        self.depth_img = np.asanyarray(depth_img)

    def set_color_img(self, color_img):
        self.color_img = np.asanyarray(color_img)

    def set_depth_img(self, depth_img):
        self.depth_img = np.asanyarray(depth_img)

    def backproject(self) -> np.ndarray:
        if self.depth_img is not None:
            depth_img = self.depth_img
        else:
            raise ValueError("No depth frame provided")
        h, w = depth_img.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        x = ((u - self.cx) * depth_img / self.fx).astype(np.float32)
        y = ((v - self.cy) * depth_img / self.fy).astype(np.float32)
        z = depth_img.astype(np.float32)

        return np.stack([x, y, z], axis=-1)

    def projet(self, points: np.ndarray) -> np.ndarray:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        u = (x * self.fx / z + self.cx).astype(np.int)
        v = (y * self.fy / z + self.cy).astype(np.int)
        return np.stack([u, v], axis=-1)

    def pixel_to_camera_frame(self, pixel: np.ndarray):
        u, v = pixel[:, 0], pixel[:, 1]
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return np.stack([x, y], axis=-1)


class RealSenseCamera(Camera):
    def __init__(
        self,
        fx,
        fy,
        cx,
        cy,
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        frame_rate=30,
    ):
        super().__init__(fx, fy, cx, cy, color_width, color_height)
        self.frame_rate = frame_rate
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.depth, color_width, color_height, rs.format.z16, frame_rate
        )
        config.enable_stream(
            rs.stream.color, depth_width, depth_height, rs.format.bgr8, frame_rate
        )
        self.pipeline.start(config)
        print("camera Started:")
        print("Width: ", color_width)
        print("Height: ", color_height)
        print("Frame Rate: ", frame_rate)
        self.align_to_color = rs.align(rs.stream.color)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_img = color_frame.get_data()
        depth_img = depth_frame.get_data()
        self.set_img(color_img, depth_img)

    def stop(self):
        self.pipeline.stop()

    def show(self):
        if self.color_img is not None:
            cv2.imshow("Color Image", self.color_img)
        if self.depth_img is not None:
            cv2.imshow("Depth Image", self.depth_img)

    def recordVideo(self, save_path):
        if not save_path.endswith(".avi"):
            raise ValueError("Only .avi format is supported")
        print("Begin recording Video, Press Q to stop")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            save_path, fourcc, self.frame_rate, (self.width, self.height)
        )

        while True:
            self.get_frame()
            out.write(self.color_img)
            self.show()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        out.release()
        print("End recording Video")
        cv2.destroyAllWindows()
