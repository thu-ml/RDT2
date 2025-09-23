import sys
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from multiprocessing.managers import SharedMemoryManager
import cv2
# from utils.img_utils import ImageViewer
from deploy.umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from deploy.umi.real_world.camera.base import BaseController, BaseControllerConfig

try:
    sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
    from MvCameraControl_class import *
except ImportError:
    print("Failed to import MvCameraControl_class. Ensure the MVS SDK is installed and the path is correct.")



@dataclass
class CamControllerConfig(BaseControllerConfig):
    receive_latency: float = 0.0
    img_transform_func: Optional[Callable] = None

    width: int = 1
    height: int = 1

    transformed_width: int = 1
    transformed_height: int = 1

    transformed_feedback_sample: Optional[Dict[str, np.ndarray]] = None
    
    output_dir: Optional[str] = None

    def validate(self):
        super().validate()
        self.feedback_sample = {
            'img': np.zeros((self.height, self.width, 3), dtype=np.uint8),
            'img_receive_timestamp': np.zeros((1,), dtype=np.float64),
            'img_timestamp': np.zeros((1,), dtype=np.float64),
        }
        self.transformed_feedback_sample = {
            'img': np.zeros((self.transformed_height, self.transformed_width, 3), dtype=np.uint8),
            'img_receive_timestamp': np.zeros((1,), dtype=np.float64),
            'img_timestamp': np.zeros((1,), dtype=np.float64),
        }

class CamController(BaseController):
    config: CamControllerConfig

    def __init__(self, config: CamControllerConfig):
        super().__init__(config)

        if config.img_transform_func is not None:
            shm_manager = SharedMemoryManager()
            shm_manager = shm_manager.__enter__()
            self.transformed_feedback_queue = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=self.config.transformed_feedback_sample,
                get_max_k=config.get_max_k,
                get_time_budget=config.get_time_budget,
                put_desired_frequency=config.put_desired_frequency,
            )

        self.last_img: np.ndarray = None
        self.last_timestamp: float = None
        
        if self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)

    ################## abstract methods ##################
    def get_transformed_feedback(self, k=None):
        if k is None:
            return self.transformed_feedback_queue.get()
        else:
            return self.transformed_feedback_queue.get_last_k(k=k)

    def _process_commands(self):
        pass

    def _initialize(self):
        pass

    def _update(self):
        assert self.last_img is not None and self.last_timestamp is not None, f"last_img or last_timestamp is None, last_img: {type(self.last_img)}, last_timestamp: {type(self.last_timestamp)}"
        self.feedback_queue.put({
            'img': self.last_img.copy(),
            'img_timestamp': np.array([self.last_timestamp - self.config.receive_latency], dtype=np.float64),
            'img_receive_timestamp': np.array([self.last_timestamp], dtype=np.float64),
        })

        if self.config.img_transform_func is not None:
            transformed_img = self.config.img_transform_func(self.last_img).copy()
            self.transformed_feedback_queue.put({
                'img': transformed_img.copy(),
                'img_timestamp': np.array([self.last_timestamp - self.config.receive_latency], dtype=np.float64),
                'img_receive_timestamp': np.array([self.last_timestamp], dtype=np.float64),
            })
        
        # if self.config.output_dir is not None:
        #     cv2.imwrite(os.path.join(self.config.output_dir, f"{self.last_timestamp}.png"), self.last_img)

    def _close(self):
        pass

    def reset(self):
        pass
