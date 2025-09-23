from typing import List, Optional, Union, Dict, Callable
import copy
import numpy as np

from deploy.umi.real_world.camera.mvs_cam import MVSCamController


class MultiMVSCamera:
    def __init__(self,
            configs
        ):
        super().__init__()

        self.cameras = [
            MVSCamController(config)
            for config in configs
        ]

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        return True
    
    def start(self, wait=True, put_start_time=None):
        for camera in self.cameras:
            camera.start()
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras:
            camera.stop()
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras:
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras:
            camera.end_wait()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get_transformed_feedback(k=k)
            out[i] = {
                'color': this_out["img"],
                "camera_receive_timestamp": this_out["img_receive_timestamp"],
                "camera_capture_timestamp": this_out["img_timestamp"],
                "timestamp": this_out["img_timestamp"]
            }
        return out

    def get_vis(self, k=None, out=None):
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get_feedback(k=k)
            out[i] = {
                'color': this_out["img"],
                "timestamp": this_out["img_timestamp"]
            }
        return out
    
    def start_recording(self, *args, **kwargs):
        pass
        
    def stop_recording(self, k=None):
        pass

    def restart_put(self, start_time):
        pass