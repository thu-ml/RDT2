import os
import time
import numpy as np
import enum
import multiprocessing as mp
from queue import Queue
from multiprocessing.managers import SharedMemoryManager
from deploy.umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from deploy.umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from deploy.umi.common.precise_sleep import precise_wait
from deploy.umi.real_world.zhixing_driver import ZhixingDriver
from deploy.umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

def is_onset_at(
    signal: np.ndarray,
    idx: int,
    *,
    jump_thr: float = 5e-5,
    range_thr: float = 0.0025,
    after_jump: int = 3,
) -> int:
    """Check whether `idx` is the first sample of a sudden change.

    An onset is the first point whose absolute jump (up or down) with respect
    to the previous sample exceeds `jump_thr`, **after** a period of at least
    `pre_quiet` consecutive "small-jump" samples.

    Args:
        signal: 1-D array of shape ``(T,)`` containing the time-series values.
        idx: Index to evaluate. Must satisfy ``1 <= idx < len(signal)``.
        jump_thr: Absolute size of a jump that counts as "sudden".  
        pre_quiet: Number of immediately preceding samples that must all have
            ``|Δ| < jump_thr`` for `idx` to qualify as an onset.

    Returns:
        Sign of onset if `idx` marks the beginning of a sudden change, otherwise 0.

    Raises:
        IndexError: If `idx` is out of bounds (``idx <= 0`` or
            ``idx >= len(signal)``).

    Notes:
        * Runs in ``O(pre_quiet)`` time - fine for interactive calls.
        * The definition is symmetric: spikes **up** and **down** are both
          considered onsets.

    Example:
        >>> sig = np.array([0.0, 0.1, 0.12, 1.0, 1.05, 1.1, 0.15, 0.12, -0.9])
        >>> is_onset_at(sig, 3)
        True
    """
    # ---------- 0. validate index ----------
    if idx <= 0 or idx >= signal.size:
        raise IndexError("idx must be between 1 and len(signal)-1")

    # ---------- 1. check if the signal is in the range ----------
    if signal.max() - signal.min() < range_thr:
        return 0
    
    jump_now = abs(signal[idx] - signal[idx - 1])
    if jump_now < jump_thr:
        return 0

    end = min(idx + after_jump, signal.size - 1)
    jump_signs = np.sign(np.diff(signal[idx - 1 : end]))
    return jump_signs[0] if (np.all(jump_signs > 0) or np.all(jump_signs < 0)) else 0
    

def get_serial_dev(serial: str):
    """
    Args:
        lr_mapping: left | right: serial
    """
    # check all devs are /dev/ttyUSB*
    devs = os.listdir('/dev')
    devs = [dev for dev in devs if dev.startswith('ttyUSB')]
    dev_ids = [int(dev.split('ttyUSB')[-1]) for dev in devs]

    for dev_id in dev_ids:
        try:
            with ZhixingDriver(serial_dev=f"/dev/ttyUSB{dev_id}") as zx:
                zx_serial = zx.read_serial()
            if zx_serial == serial:
                return f"/dev/ttyUSB{dev_id}"
        except Exception as e:
            # print(f"Error reading serial from /dev/ttyUSB{dev_id}: {e}")
            continue

    raise ValueError(f"Serial {serial} not found")


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class ZhixingController(mp.Process):
    def __init__(self,
        shm_manager: SharedMemoryManager,
        serial,
        baud=115200,
        frequency=15,
        get_max_k=None,
        command_queue_size=1024,
        launch_timeout=3,
        receive_latency=0.0,
        open_width=0.12,  # unit: m
        closed_width=0.0,  # unit: m
        force=30,
        verbose=False
    ):
        super().__init__(name="ZhixingController")
        self.serial = serial
        self.serial_dev = get_serial_dev(serial)
        print(f"[DBG] serial: {serial}, serial_dev: {self.serial_dev}")
        
        self.baud = baud
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        self.open_width = open_width
        self.closed_width = closed_width
        self.force = force

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_position': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time(),
            'gripper_reached': False
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.waypoint_queue = Queue(maxsize=1000)

    def width_to_pos(self, width):
        """
        Convert width (m) to corresponding pos (0-1.0), 1.0 = open.
        """
        return ((width - self.closed_width) 
                / (self.open_width - self.closed_width))
    
    def pos_to_width(self, pos):
        """
        Convert pos (0-1.0) to corresponding width (m), 1.0 = open.
        """
        return (pos * (self.open_width - self.closed_width) 
                + self.closed_width)
    

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ZhixingController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)


    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # start connection
        try:
            with ZhixingDriver(
                serial_dev=self.serial_dev, 
                baud=self.baud) as zx:

                if self.force is not None:
                    zx.set_force(self.force)
                
                # get initial
                curr_pos = zx.read_pos()
                curr_t = time.monotonic()
                last_waypoint_time = curr_t
                pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[[curr_pos,0,0,0,0,0]]
                )
                
                # TODO(bangguo): apply filter here to ensure smooth motion
                # Actually I do not whether it's necessary
                # cause we need to detct the onset
                # we do not want the perturbation to be too large to be detected as onset
                keep_running = True
                t_start = time.monotonic()
                iter_idx = 0
                while keep_running:
                    # command gripper
                    t_now = time.monotonic()
                    dt = 1 / self.frequency
                    
                    # if t_now is on the begining of an onset, then send signal
                    # sampled_timestamps = np.arange(
                    #     max(t_now - 2 * dt, pose_interp.times[0]), 
                    #     t_now + 0.8, # extrapolation may happen here
                    #     step=dt
                    # )   # TODO(lingxuan): remove magic number 2*dt
                    # sampled_widths = pose_interp(sampled_timestamps)[..., 0]
                    # print(f"[DBG] {sampled_widths}")
                    # onset_sign = is_onset_at(
                    #     signal=sampled_widths,
                    #     idx=2,  # TODO(lingxuan): remove magic number 2 (index of t_now)
                    # )   # TODO(lingxuan): load hyper params from config
                    # if onset_sign != 0:
                    #     # NOTE: only binarize the low pmw
                    #     curr_pos = zx.read_pos()
                    #     target_pos = pose_interp(t_now + 1.6)[0]
                    #     # print(f"[{self.serial}] curr_width: {self.pos_to_width(curr_pos)}, target_width: {self.pos_to_width(target_pos)}")
                    #     # print(f"[DBG] t_now: {t_now - pose_interp.times[0]},"
                    #     #       f" curr_pos: {curr_pos}, target_pos: {target_pos}"
                    #     #       f" widths: {sampled_widths}")
                    #     if (onset_sign == np.sign(target_pos - curr_pos) 
                    #         and abs(target_pos - curr_pos) > 0.005
                    #     ):
                    #         # moving direction matched with the onset
                    #         # target_pos = target_pos if target_pos > curr_pos else target_pos
                    #         # target_pos = target_pos if target_pos > curr_pos else 0.0
                    #         # move with block to later actuation while executing previous command
                    #         # print(f"[{self.serial}] move to target width: {self.pos_to_width(target_pos)}")
                    #         zx.move_and_wait_for_pos(target_pos)
                    
                    first_waypoint_in_queue = self.waypoint_queue.get_nowait() if not self.waypoint_queue.empty() else None
                    if first_waypoint_in_queue is not None:
                        target_time = first_waypoint_in_queue['target_time']
                        # while (target_time < t_now + 0.05):
                        #     # print(f"[{self.serial}] target_time {target_time} is too early, skip")
                        #     first_waypoint_in_queue = self.waypoint_queue.get_nowait() if not self.waypoint_queue.empty() else None
                        #     if first_waypoint_in_queue is None:
                        #         break
                        #     target_time = first_waypoint_in_queue['target_time']
                        # Now the first_waypoint_in_queue is the first waypoint that is not too early
                        if first_waypoint_in_queue is not None:
                            target_pos = first_waypoint_in_queue['target_pos']
                            target_time = first_waypoint_in_queue['target_time']
                            #       f" curr_pos: {curr_pos}, target_pos: {target_pos}")
                            if abs(target_pos - curr_pos) > 0.005:
                                # move with block to later actuation while executing previous command
                                zx.move_to(normalized_pos=target_pos)
                                last_waypoint_time = target_time

                    pos = zx.read_pos()
                    
                    # get state from robot
                    state = {
                        'gripper_position': self.pos_to_width(pos),
                        'gripper_receive_timestamp': time.time(),
                        'gripper_timestamp': time.time() - self.receive_latency,
                        'gripper_reached': zx.is_reached()
                    }
                    self.ring_buffer.put(state)

                    # fetch command from queue
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0
                    
                    # execute commands
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd']
                        
                        if cmd == Command.SHUTDOWN.value:
                            keep_running = False
                            # stop immediately, ignore later commands
                            break
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_width = command['target_pos']
                            target_pos = self.width_to_pos(target_width)
                            target_time = command['target_time']
                            
                            # translate global time to monotonic time
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now
                            # print('controller ', target_pos, target_width, target_time)
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[target_pos, 0, 0, 0, 0, 0],
                                time=target_time,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            self.waypoint_queue.put({
                                'target_pos': target_pos,
                                'target_time': target_time
                            })
                            last_waypoint_time = target_time
                        elif cmd == Command.RESTART_PUT.value:
                            t_start = command['target_time'] - time.time() + time.monotonic()
                            iter_idx = 1
                        else:
                            keep_running = False
                            break
                        
                    # first loop successful, ready to receive command
                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx += 1
                    
                    # regulate frequency
                    dt = 1 / self.frequency
                    t_end = t_start + dt * iter_idx
                    precise_wait(t_end=t_end, time_func=time.monotonic)
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[ZhixingController] Disconnected from robot: {self.serial_dev}")