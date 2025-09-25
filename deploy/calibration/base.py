from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np
import time
import copy
import traceback
import logging
import threading

from deploy.umi.shared_memory import (
    SharedMemoryQueue, SharedMemoryRingBuffer, SharedMemoryManager
)
from deploy.umi.common.precise_sleep import precise_wait, precise_sleep


@dataclass
class BaseControllerConfig:
    """
    Base configuration class for all controllers.
    """
    name: str = ""
    fps: Optional[int] = None
    put_desired_frequency: int = None
    command_sample: Optional[Dict] = None
    feedback_sample: Optional[Dict[str, np.ndarray]] = None

    buffer_size: int = 1000
    get_max_k: int = 10
    get_time_budget: float = 1.0

    precision_wait_slack_time: float = 0.0001
    precision_wait_time_func: callable = time.monotonic

    verbose: bool = False

    def validate(self):
        assert isinstance(self.name, str) and len(self.name) > 0, f"Invalid name: {self.name}"
        assert self.put_desired_frequency is not None, "put_desired_frequency is None"

class BaseController(mp.Process, ABC):
    """
    Abstract base class for all robot controllers running as separate processes.
    Provides common functionality for initialization, control, and error handling.
    """

    def __init__(self, config: BaseControllerConfig):
        """
        Initialize the controller with the given configuration.

        Args:
            config: Controller configuration object
            command_queue: Queue for receiving commands from parent process
            feedback_queue: Queue for sending feedback to parent process
        """
        super().__init__()

        # Validate the configuration
        config.validate()
        self.config = config

        if config.command_sample is not None:
            shm_manager = SharedMemoryManager()
            shm_manager = shm_manager.__enter__()
            self.command_queue = SharedMemoryQueue.create_from_examples(
                shm_manager=shm_manager,
                examples=config.command_sample,
                buffer_size=config.buffer_size,
            )

        if config.feedback_sample is not None:
            shm_manager = SharedMemoryManager()
            shm_manager = shm_manager.__enter__()
            self.feedback_queue = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=config.feedback_sample,
                get_max_k=config.get_max_k,
                get_time_budget=config.get_time_budget,
                put_desired_frequency=config.put_desired_frequency,
            )

        self.mp_manager = mp.Manager()

        self._ready_event = mp.Event()
        self._stop_event = mp.Event()
        self._logger = logging.getLogger(f"{self.__class__.__name__}_{self.config.name}")

        self.fps = self.config.fps
        if self.fps is not None:
            self.dt = 1.0 / self.fps
        else:
            self.dt = None

        self.daemon = True

    def run(self):
        """
        Main process loop.
        """
        try:
            print(f"initializing {self.config.name} controller")
            self._initialize()
            print(f"resetting {self.config.name} controller")
            self.reset()
            self._ready_event.set()
            print(f"{self.config.name} controller initialized successfully, ready to receive commands")
            # self._logger.info(f"{self.config.name} controller initialized successfully")

            t_start = time.monotonic()
            iter_idx = 0
            while not self._stop_event.is_set():
                if self.dt is not None:
                    t_wait_util = t_start + (iter_idx + 1) * self.dt

                # Process incoming commands
                self._process_commands()

                # Perform controller update
                self._update()

                # Sleep to maintain control frequency
                if self.dt is not None:
                    precise_wait(t_wait_util, slack_time=self.config.precision_wait_slack_time, time_func=self.config.precision_wait_time_func)

                iter_idx += 1
        except Exception as e:
            traceback.print_exc()
            self._stop_event.set()
            self._logger.error(f"Controller {self.config.name} crashed: {str(e)}", exc_info=True)

        try:
            self._close()
        except Exception as e:
            traceback.print_exc()
            self._logger.error(f"Error during closing {self.config.name} controller: {str(e)}", exc_info=True)
        finally:
            self._logger.info(f"{self.config.name} controller stopped")

    def check_alive(self):
        return not self._stop_event.is_set()

    @abstractmethod
    def _process_commands(self):
        """
        Process all pending commands from the command queue.
        """
        pass

    @abstractmethod
    def _initialize(self):
        """
        Controller-specific initialization to be implemented by child classes.
        """
        pass

    @abstractmethod
    def _update(self):
        """
        Controller-specific update loop to be implemented by child classes.
        This is where the actual control logic goes.
        """
        pass

    @abstractmethod
    def _close(self):
        """
        Controller-specific cleanup to be implemented by child classes.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the controller state.
        This method is intended to be overridden by subclasses if needed.
        """
        pass

    def stop(self):
        """
        Signal the controller to stop gracefully.
        """
        self._stop_event.set()

    def get_feedback(self):
        return self.feedback_queue.get()

    def send_command(self, command):
        """
        Send a command to the controller process (to be called from parent process).

        Args:
            command: Name of the command/method to execute
            *args: Arguments for the command
        """
        self.command_queue.put(command)

    def __enter__(self):
        """Support for context manager protocol."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager protocol."""
        self.stop()
        self.join()
