import os
import time
import threading
from typing import Optional

import cv2
import numpy as np

from .stats import StreamStats

class CameraStream:
    """
    Захватывает кадры с камеры в отдельном потоке.
    Хранит только последний кадр.
    """
    def __init__(self, device:str, width: int = 854, height:int = 480, fps: int = 10):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None

        self.stats = StreamStats()
    
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        
        self._init_cap()

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        
        self._cap = None
        self._thread = None

    def get_last_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()
        
    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self.stats.frames_drop += 1
                time.sleep(0.01)
                continue

            with self._lock:
                self._last_frame = frame
                self.stats.frames_ok += 1
                self.stats.last_frame_ts = time.time()

    def _init_cap(self) -> None:
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

def get_device_from_env() -> str:
    return os.environ.get("VIDEO_DEVICE", "dev/video2")

