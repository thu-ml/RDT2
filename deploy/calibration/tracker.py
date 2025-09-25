from __future__ import annotations
from deploy.calibration.base import BaseControllerConfig, BaseController
from dataclasses import field, dataclass
from typing import Optional, Any, Dict
import enum
import multiprocessing as mp
import time
import traceback
import os
import copy
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R



import time
import sys
import openvr
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

__all__ = ["ViveTrackerModule", "ViveTrackerUpdater", "get_all_tracker_devices", "get_all_tracker_serial_no"]


EYE_T = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    float,
)

def eye_T():
    return EYE_T.copy()

def matrix_to_pos_quat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 提取位置
    pos = T[:3, 3]

    # 提取旋转矩阵
    rotation_matrix = T[:3, :3]

    # 使用 scipy 将旋转矩阵转换为四元数
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # [x, y, z, w]

    return pos, quat

def quaternion_rotation_distance(q1, q2):
    # q1, q2: [x, y, z, w]
    # 转换为四元数对象
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    # 计算相对旋转四元数 q_rel = q2 * q1^-1
    relative_rotation = r2 * r1.inv()

    # 提取旋转角度，单位是弧度
    angle_rad = relative_rotation.magnitude()

    # 转换为度
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def get_all_tracker_devices(device_key="tracker"):
    vive_tracker_module = ViveTrackerModule()
    vive_tracker_module.print_discovered_objects()
    tracking_devices = vive_tracker_module.return_selected_devices(device_key)
    tracking_devices_keys = list(tracking_devices.keys())

    return tracking_devices_keys

def get_all_tracker_serial_no():
    vive_tracker_module = ViveTrackerModule()
    vive_tracker_module.print_discovered_objects()
    tracking_devices = vive_tracker_module.return_selected_devices("tracker")
    tracking_devices_keys = list(tracking_devices.keys())

    serial_no_list = []
    for key in tracking_devices_keys:
        serial_no_list.append(tracking_devices[key].get_serial())

    return serial_no_list

class ViveTrackerUpdater():
    def __init__(self, calc_test_dist_duration_time: None | int = None, device = None, device_series_no = None):
        '''
        Args:
            device: tracker_1, tracker_2, ...
            device_series_no
        '''
        if calc_test_dist_duration_time is not None:
            self.calc_test_dist_duration_time = int(calc_test_dist_duration_time)
        else:
            self.calc_test_dist_duration_time = None

        self.vive_tracker_module = ViveTrackerModule()

        # self.vive_tracker_module.print_discovered_objects()

        self.device_key = "tracker"
        self.tracking_devices = self.vive_tracker_module.return_selected_devices(self.device_key)

        if device is not None:
            self.tracking_devices = {
                device: self.tracking_devices[device]
            }

        if device_series_no is not None:
            tgt_device = None

            # get_serial
            for key in self.tracking_devices:
                # print(self.tracking_devices[key].get_serial())
                if device_series_no in self.tracking_devices[key].get_serial():
                    tgt_device = key
                    break

            if tgt_device is not None:
                self.tracking_devices = {
                    tgt_device: self.tracking_devices[tgt_device]
                }
            else:
                print(f"No device which has {device_series_no} in serial.")
                self.tracking_devices_keys = []
                return

        self.tracking_devices_keys = list(self.tracking_devices.keys())

        print(f"Tracking devices: {self.tracking_devices_keys}")
        if len(self.tracking_devices) == 0:
            print(f"No devices found.")
            # exit(0)
        elif len(self.tracking_devices) == 1:
            print(f"Use one device.")
        else:
            # user_select_device_idx = input(f"Select a device index from 0 to {len(self.tracking_devices) - 1}: ")
            # self.tracking_devices = {
            #     self.tracking_devices_keys[int(user_select_device_idx)]: self.tracking_devices[self.tracking_devices_keys[int(user_select_device_idx)]]
            # }
            pass
        self.tracking_result = {}

        self.prev_tracking_result = None
        self._init_center_pos = None

        if self.calc_test_dist_duration_time is not None:
            self.pos_history = {key: [] for key in self.tracking_devices}
            self.quat_history = {key: [] for key in self.tracking_devices}

    def update(self, verbose=False):
        new_tracking_result = {}

        for key in self.tracking_devices:
            # x, y, z, qx, qy, qz, qw
            cur_pos, cur_quat = matrix_to_pos_quat(self.tracking_devices[key].get_T())
            # print(f"key: {key}, cur_pos: {cur_pos}")
            new_tracking_result[key] = {
                "pos": cur_pos,
                "quat": cur_quat
            }

        if self._init_center_pos is None:
            self._init_center_pos = deepcopy(sum([new_tracking_result[key]["pos"] for key in new_tracking_result]) / len(new_tracking_result))

        for key in list(new_tracking_result.keys()):
            cur_pos, cur_quat = new_tracking_result[key]["pos"], new_tracking_result[key]["quat"]
            # cur_pos -= self._init_center_pos

            if self.prev_tracking_result is not None and \
                    len(self.prev_tracking_result) > 0:
                prev_pos, prev_quat = self.prev_tracking_result[key]["pos"], self.prev_tracking_result[key]["quat"]
                distance = np.linalg.norm(cur_pos - prev_pos)

                # print(f"distance: {distance}, max_dist: {self.max_dist}, cur_pos: {cur_pos}, prev_pos: {prev_pos}")

                # if distance > self.max_dist:
                #     # cur_pos = prev_pos

                #     direction = (cur_pos - prev_pos) / distance
                #     cur_pos = prev_pos + self.max_dist * direction

            new_tracking_result[key] = {
                "pos": cur_pos,
                "quat": cur_quat
            }

        self.prev_tracking_result = deepcopy(self.tracking_result)
        self.tracking_result = new_tracking_result

        # print(self.prev_tracking_result, self.tracking_result)

        if verbose:
            for r in self.tracking_result:
                print("\r" + str(r), end="")

class ViveTrackerModule():
    def __init__(self, configfile_path=None):
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()
        self.object_names = {"Tracking Reference":[],"HMD":[],"Controller":[],"Tracker":[]}
        self.devices = {}
        self.device_index_map = {}
        poses = self.vr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0,
                                                               openvr.k_unMaxTrackedDeviceCount)
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bDeviceIsConnected:
                self.add_tracked_device(i)

    def __del__(self):
        openvr.shutdown()

    def return_selected_devices(self, device_key=""):
        selected_devices = {}
        for key in self.devices:
            if device_key in key:
                selected_devices[key] = self.devices[key]
        return selected_devices

    def get_pose(self):
        return get_pose(self.vr)

    def poll_vr_events(self):
        event = openvr.VREvent_t()
        while self.vrsystem.pollNextEvent(event):
            if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                self.add_tracked_device(event.trackedDeviceIndex)
            elif event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                if event.trackedDeviceIndex in self.device_index_map:
                    self.remove_tracked_device(event.trackedDeviceIndex)

    def add_tracked_device(self, tracked_device_index):
        i = tracked_device_index
        device_class = self.vr.getTrackedDeviceClass(i)
        if (device_class == openvr.TrackedDeviceClass_Controller):
            device_name = "controller_"+str(len(self.object_names["Controller"])+1)
            self.object_names["Controller"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr,i,"Controller")
            self.device_index_map[i] = device_name
        elif (device_class == openvr.TrackedDeviceClass_HMD):
            device_name = "hmd_"+str(len(self.object_names["HMD"])+1)
            self.object_names["HMD"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr,i,"HMD")
            self.device_index_map[i] = device_name
        elif (device_class == openvr.TrackedDeviceClass_GenericTracker):
            device_name = "tracker_"+str(len(self.object_names["Tracker"])+1)
            self.object_names["Tracker"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr,i,"Tracker")
            self.device_index_map[i] = device_name
        elif (device_class == openvr.TrackedDeviceClass_TrackingReference):
            device_name = "tracking_reference_"+str(len(self.object_names["Tracking Reference"])+1)
            self.object_names["Tracking Reference"].append(device_name)
            self.devices[device_name] = vr_tracking_reference(self.vr,i,"Tracking Reference")
            self.device_index_map[i] = device_name

    def remove_tracked_device(self, tracked_device_index):
        if tracked_device_index in self.device_index_map:
            device_name = self.device_index_map[tracked_device_index]
            self.object_names[self.devices[device_name].device_class].remove(device_name)
            del self.device_index_map[tracked_device_index]
            del self.devices[device_name]
        else:
            raise Exception("Tracked device index {} not valid. Not removing.".format(tracked_device_index))

    def rename_device(self,old_device_name,new_device_name):
        self.devices[new_device_name] = self.devices.pop(old_device_name)
        for i in range(len(self.object_names[self.devices[new_device_name].device_class])):
            if self.object_names[self.devices[new_device_name].device_class][i] == old_device_name:
                self.object_names[self.devices[new_device_name].device_class][i] = new_device_name

    def print_discovered_objects(self):
        for device_type in self.object_names:
            plural = device_type
            if len(self.object_names[device_type])!=1:
                plural+="s"
            print("Found "+str(len(self.object_names[device_type]))+" "+plural)
            for device in self.object_names[device_type]:
                if device_type == "Tracking Reference":
                    print("  "+device+" ("+self.devices[device].get_serial()+
                          ", Mode "+self.devices[device].get_model()+
                          ", "+self.devices[device].get_model()+
                          ")")
                else:
                    print("  "+device+" ("+self.devices[device].get_serial()+
                          ", "+self.devices[device].get_model()+")")

def update_text(txt):

    """Update the text in the same line on the console.

    Args:
        txt (str): The text to display.
    """
    sys.stdout.write('\r' + txt)
    sys.stdout.flush()

def convert_to_euler(pose_mat):
    """Convert a 3x4 position/rotation matrix to an x, y, z location and the corresponding Euler angles (in degrees).

    Args:
        pose_mat (list): A 3x4 position/rotation matrix.

    Returns:
        list: A list containing x, y, z, yaw, pitch, and roll values.
    """
    yaw = 180 / math.pi * math.atan2(pose_mat[1][0], pose_mat[0][0])
    pitch = 180 / math.pi * math.atan2(pose_mat[2][0], pose_mat[0][0])
    roll = 180 / math.pi * math.atan2(pose_mat[2][1], pose_mat[2][2])
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x, y, z, yaw, pitch, roll]

def convert_to_quaternion(pose_mat):

    """Convert a 3x4 position/rotation matrix to an x, y, z location and the corresponding quaternion.

    Args:
        pose_mat (list): A 3x4 position/rotation matrix.

    Returns:
        list: A list containing x, y, z, r_w, r_x, r_y, and r_z values.
    """
    # Calculate quaternion values
    r_w = math.sqrt(abs(1 + pose_mat[0][0] + pose_mat[1][1] + pose_mat[2][2])) / 2
    r_x = (pose_mat[2][1] - pose_mat[1][2]) / (4 * r_w)
    r_y = (pose_mat[0][2] - pose_mat[2][0]) / (4 * r_w)
    r_z = (pose_mat[1][0] - pose_mat[0][1]) / (4 * r_w)

    # Get position values
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]

    return [x, y, z, r_w, r_x, r_y, r_z]

#Define a class to make it easy to append pose matricies and convert to both Euler and Quaternion for plotting
class pose_sample_buffer():
    def __init__(self):
        self.i = 0
        self.index = []
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.r_w = []
        self.r_x = []
        self.r_y = []
        self.r_z = []

    def append(self,pose_mat,t):
        self.time.append(t)
        self.x.append(pose_mat[0][3])
        self.y.append(pose_mat[1][3])
        self.z.append(pose_mat[2][3])
        self.yaw.append(180 / math.pi * math.atan(pose_mat[1][0] /pose_mat[0][0]))
        self.pitch.append(180 / math.pi * math.atan(-1 * pose_mat[2][0] / math.sqrt(pow(pose_mat[2][1], 2) + math.pow(pose_mat[2][2], 2))))
        self.roll.append(180 / math.pi * math.atan(pose_mat[2][1] /pose_mat[2][2]))
        r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
        self.r_w.append(r_w)
        self.r_x.append((pose_mat[2][1]-pose_mat[1][2])/(4*r_w))
        self.r_y.append((pose_mat[0][2]-pose_mat[2][0])/(4*r_w))
        self.r_z.append((pose_mat[1][0]-pose_mat[0][1])/(4*r_w))

def get_pose(vr_obj):

    """Get the pose of a tracked device in the virtual reality system.

    Args:
        vr_obj (openvr object): An instance of the openvr object.

    Returns:
        list: A list of poses for each tracked device in the system.
    """
    return vr_obj.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)


class vr_tracked_device():

    def __init__(self, vr_obj, index, device_class):
        self.device_class = device_class
        self.index = index
        self.vr = vr_obj
        self.T = eye_T()


    def get_serial(self):
        """Get the serial number of the tracked device."""
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_SerialNumber_String)

    def get_model(self):
        """Get the model number of the tracked device."""
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_ModelNumber_String)

    def get_battery_percent(self):
        """Get the battery percentage of the tracked device."""
        return self.vr.getFloatTrackedDeviceProperty(self.index, openvr.Prop_DeviceBatteryPercentage_Float)

    def is_charging(self):
        """Check if the tracked device is charging."""
        return self.vr.getBoolTrackedDeviceProperty(self.index, openvr.Prop_DeviceIsCharging_Bool)


    def sample(self, num_samples, sample_rate):
        """Sample the pose of the tracked device.

        Args:
            num_samples (int): Number of samples to collect.
            sample_rate (float): Rate at which to collect samples.

        Returns:
            PoseSampleBuffer: A buffer containing the collected pose samples.
        """
        interval = 1 / sample_rate
        rtn = pose_sample_buffer()
        sample_start = time.time()
        for i in range(num_samples):
            start = time.time()
            pose = get_pose(self.vr)
            rtn.append(pose[self.index].mDeviceToAbsoluteTracking, time.time() - sample_start)
            sleep_time = interval - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
        return rtn

    def get_T(self, pose=None):
        pose_mat = self.get_pose_matrix()
        if pose_mat: # not None
            np_pose_mat = np.array(pose_mat)['m']
            self.T[:3,:] = np_pose_mat
        return self.T

    def get_pose_euler(self, pose=None):
        """Get the pose of the tracked device in Euler angles.

        Args:
            pose (list, optional): The current pose of the device. If not provided, get_pose is called.

        Returns:
            tuple: Euler angles representing the pose, or None if the pose is not valid.
        """
        if pose is None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_euler(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def get_pose_matrix(self, pose=None):
        """Get the pose matrix of the tracked device.

        Args:
            pose (list, optional): The current pose of the device. If not provided, get_pose is called.

        Returns:
            list: The pose matrix of the device, or None if the pose is not valid.
        """
        if pose is None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].mDeviceToAbsoluteTracking
        else:
            return None

    def get_velocity(self, pose=None):
        """Get the linear velocity of the tracked device.

        Args:
            pose (list, optional): The current pose of the device. If not provided, get_pose is called.

        Returns:
            tuple: The linear velocity of the device, or None if the pose is not valid.
        """
        if pose is None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vVelocity
        else:
            return None

    def get_angular_velocity(self, pose=None):
        # Get the angular velocity of the tracked device if its pose is valid.
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vAngularVelocity
        else:
            return None

    def get_pose_quaternion(self, pose=None):
        # Get the pose of the tracked device in the form of a quaternion if its pose is valid.
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_quaternion(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def controller_state_to_dict(self, pControllerState):
        # Convert controller state data to a dictionary for easier use.
        d = {}
        # Fill dictionary with controller state data
        ...
        return d

    def get_controller_inputs(self):
        # Get the current state of the controller inputs.
        result, state = self.vr.getControllerState(self.index)
        return self.controller_state_to_dict(state)

    def trigger_haptic_pulse(self, duration_micros=1000, axis_id=0):
        # Trigger a haptic pulse on the controller.
        self.vr.triggerHapticPulse(self.index ,axis_id, duration_micros)

class vr_tracking_reference(vr_tracked_device):
    def get_mode(self):
        # Get the mode of the tracking reference.
        return self.vr.getStringTrackedDeviceProperty(self.index,openvr.Prop_ModeLabel_String).decode('utf-8').upper()

    def sample(self,num_samples,sample_rate):
        # Warn the user that sampling a tracking reference is not useful, as they do not move.
        print("Tracker static!")




@dataclass
class TrackerControllerConfig(BaseControllerConfig):
    receive_latency: float = 0.0

    tracker_names: list[str] = field(default_factory=list)
    tracker_serials: list[str] = field(default_factory=list)

    not_ready_return_to_init_dev_wait_time: float = 10.0
    not_ready_sleep_time: float = 0.1

    tracker_check_hist: float = 0.5
    idle_time: float = 30.0
    idle_pos_eps: float = 0.005

    num_devs: int = None

    feedback_sample: Dict[str, Any] = None

    def validate(self):
        super().validate()

        assert len(self.tracker_names) == len(self.tracker_serials), \
            f"Number of tracker names {len(self.tracker_names)} does not match number of serials {len(self.tracker_serials)}"

        assert len(set(self.tracker_names)) == len(self.tracker_names), \
            f"Tracker names must be unique, found duplicates: {self.tracker_names}"
        assert len(set(self.tracker_serials)) == len(self.tracker_serials), \
            f"Tracker serials must be unique, found duplicates: {self.tracker_serials}"

        self.num_devs = len(self.tracker_names)

        self.feedback_sample = {}
        for tracker_name, tracker_serial in zip(self.tracker_names, self.tracker_serials):
            self.feedback_sample[f"{tracker_name}_pose"] = np.zeros((7,), dtype=np.float64)
            self.feedback_sample[f"{tracker_name}_receive_timestamp"] = np.zeros((1,), dtype=np.float64)
            self.feedback_sample[f"{tracker_name}_timestamp"] = np.zeros((1,), dtype=np.float64)


def get_all_tracker_serials():
    """
    Get a list of all available tracker serial numbers connected to the system.

    Returns:
        list[str]: A list of tracker serial numbers
    """
    try:
        # Create a temporary tracker updater to access connected devices
        tracker = ViveTrackerUpdater()

        # Get the tracking devices and their serial numbers
        tracker_devices = tracker.tracking_devices_keys
        tracker_serials = [tracker.vive_tracker_module.devices[device].get_serial()
                           for device in tracker_devices]

        # Clean up the tracker instance
        del tracker

        print(f"Found {len(tracker_serials)} trackers with serials: {tracker_serials}")
        return tracker_serials

    except Exception as e:
        print(f"Error getting tracker serials: {e}")
        return []

class TrackerController(BaseController):
    config: TrackerControllerConfig

    def __init__(self, config: TrackerControllerConfig):
        super().__init__(config)

        self.vive_tracker_updater: Optional[ViveTrackerUpdater] = None

        self._is_idle = self.mp_manager.Value("b", False)
        self._is_ready = self.mp_manager.Value("b", False)

    ################## cls methods ##################
    def _check_is_idle(self):
        if self.cur_is_idle:
            self.cur_idle_steps += 1
        else:
            self.cur_idle_steps = 0

        self._is_idle.value = (self.cur_idle_steps >= self.idle_steps)

    def _check_ready(self):
        if len(self.history_poss) == 0:
            ready = False
        else:
            ready = True
            for tracker_idx in range(self.config.num_devs):

                pos = self.poss[tracker_idx]
                quat = self.quats[tracker_idx]
                hist_pos = self.history_poss[0][tracker_idx]
                hist_quat = self.history_quats[0][tracker_idx]

                # True is ready, False is not ready
                pos_at_zero_check = np.sum(np.array(pos) ** 2) != 0
                pos_change_check = np.sum(np.abs(np.array(hist_pos) - np.array(pos))) != 0

                ready = ready and pos_at_zero_check and pos_change_check

        self._is_ready.value = ready

    def check_ready(self):
        return self._is_ready.value and not self._stop_event.is_set()

    def is_idle(self):
        return self._is_idle.value

    ################## abstract methods ##################
    def _initialize(self):
        self.tracker = ViveTrackerUpdater()

        while len(self.tracker.tracking_devices_keys) != self.config.num_devs:
            print(f"not enough trackers found, {len(self.tracker.tracking_devices_keys)}/{self.config.num_devs}, retrying...")
            del self.tracker
            self.tracker = ViveTrackerUpdater()

        self.tracker_devices: list[str] = self.tracker.tracking_devices_keys
        print(f"Tracker devices: {self.tracker_devices}")
        self.tracker_serials: list[str] = [self.tracker.vive_tracker_module.devices[device].get_serial() for device in self.tracker_devices]

        self.tracker_names: list[str] = [self.config.tracker_names[self.config.tracker_serials.index(serial)] for serial in self.tracker_serials]

        for config_tracker_serial in self.config.tracker_serials:
            assert config_tracker_serial in self.tracker_serials, \
                f"Tracker serial {config_tracker_serial} not found in connected devices: {self.tracker_serials}"

        self.cur_is_idle = False
        self.cur_idle_steps = 0

        # stop collecting data if tracker is idle for self.idle_steps_threshold steps
        self.idle_steps = self.config.idle_time * self.fps
        # move 0.01m in 0.5s will be considered as not idle
        self.idle_pos_eps = self.config.idle_pos_eps
        self.tracker_check_steps = int(self.config.tracker_check_hist * self.fps)

        self.history_poss = deque(maxlen=self.tracker_check_steps)
        self.history_quats = deque(maxlen=self.tracker_check_steps)

        self.sec = None
        self.poss = [[0.0, 0.0, 0.0]] * self.config.num_devs
        self.quats = [[0.0, 0.0, 0.0, 1.0]] * self.config.num_devs

    def _update(self):
        self.tracker.update()
        self.sec = time.time()

        self.history_poss.append(copy.deepcopy(self.poss))
        self.history_quats.append(copy.deepcopy(self.quats))

        this_is_idle = True

        for tracker_idx in range(self.config.num_devs):
            pos = copy.deepcopy(self.tracker.tracking_result[self.tracker_devices[tracker_idx]]["pos"])
            quat = copy.deepcopy(self.tracker.tracking_result[self.tracker_devices[tracker_idx]]["quat"])

            # old one
            hist_pos = self.history_poss[0][tracker_idx]
            hist_quat = self.history_quats[0][tracker_idx]

            hist_dist = np.linalg.norm(np.array(hist_pos) - np.array(pos))
            this_is_idle &= (hist_dist < self.idle_pos_eps)

            self.poss[tracker_idx] = pos
            self.quats[tracker_idx] = quat

        self.cur_is_idle = this_is_idle
        self._check_is_idle()
        self._check_ready()

        state_dict = {}
        for tracker_idx in range(self.config.num_devs):
            pos = self.poss[tracker_idx]
            quat = self.quats[tracker_idx]
            sec = self.sec

            pose = np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)

            state_dict[f"{self.tracker_names[tracker_idx]}_pose"] = pose
            state_dict[f"{self.tracker_names[tracker_idx]}_receive_timestamp"] = np.array([sec], dtype=np.float64)
            state_dict[f"{self.tracker_names[tracker_idx]}_timestamp"] = np.array([sec], dtype=np.float64)

        self.feedback_queue.put(state_dict)

    def stop(self):
        self.end_collect()

        super().stop()

    def _process_commands(self):
        pass

    def _close(self):
        del self.tracker
        self.tracker = None

    def reset(self):
        pass
