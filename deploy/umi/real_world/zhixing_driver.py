import minimalmodbus
import numpy as np

class ZhixingDriver:
    POSITION_HIGH_8_REG = 0x0102  # 位置寄存器高八位
    POSITION_LOW_8_REG = 0x0103  # 位置寄存器低八位
    SPEED_REG = 0x0104
    FORCE_REG = 0x0105
    MOTION_TRIGGER_REG = 0x0108
    ID_REG = 0x0810

    POS_ARR_REG = 0x0602
    FORCE_ARR_REG = 0x0601
    
    FEEDBACK_POS_HIGH_8_REG = 0x0609
    FEEDBACK_POS_LOW_8_REG = 0x060A

    MAX_POS = 0
    MIN_POS = 12000
    
    MAX_FEEDBACK_POS = 3000
    OPEN_FEEDBACK_POS = 0
    CLOSE_FEEDBACK_POS = 3000

    def __init__(self, serial_dev, baud=115200):
        self.serial_dev = serial_dev
        self.baud = baud
    
    def start(self):
        self.instrument = minimalmodbus.Instrument(self.serial_dev, 1)
        self.instrument.debug = False
        self.instrument.serial.baudrate = self.baud
        self.instrument.serial.timeout = 1

        self._last_cmd_start_width = None
        # self.lock = threading.Lock()
    
    def stop(self):
        pass
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop

    def write_position_high8(self, value):
         self.instrument.write_register(self.POSITION_HIGH_8_REG, value, functioncode=6)

    def write_position_low8(self, value):
        self.instrument.write_register(self.POSITION_LOW_8_REG, value, functioncode=6)

    def write_position(self, value):
        self.instrument.write_long(self.POSITION_HIGH_8_REG, value)

    def trigger_motion(self):
        self.instrument.write_register(self.MOTION_TRIGGER_REG, 1, functioncode=6)

    def read_pos_arr(self):
        pos_arr = self.instrument.read_long(self.POS_ARR_REG)
        return pos_arr
    
    def read_force_arr(self):
        force_arr = self.instrument.read_long(self.FORCE_ARR_REG)
        return force_arr
    
    def read_force(self):
        force = self.instrument.read_register(self.FORCE_REG, functioncode=3)
        return force
    
    def set_force(self, force: float):
        self.instrument.write_register(self.FORCE_REG, force, functioncode=6)

    def is_reached(self):
        return self.read_pos_arr() or self.read_force_arr()
    
    def move_and_wait_for_pos(self, normalized_pos: float):
        # CAUTIOUS: place the force check before to enable early return to reduce latency
        if (not self.read_force_arr() 
            and not self.read_pos_arr()
        ):
            return
        
        # print(f"[zhixing] move to {normalized_pos * 0.12}")
        self.move_to(normalized_pos)


    def move_to(self, normalized_pos: float):
        """
        Move the gripper to the specified normalized position with ~0.0305s.

        Args:
            normalized_pos (float): The normalized gripper width in range [0,1], where 0 represents fully 
                closed and 1 represents fully open position.

        Returns:
        """
        normalized_pos = np.clip(normalized_pos, 0, 1)
        # value in [0, 1]
        pos = int(normalized_pos * (self.MAX_POS - self.MIN_POS) + self.MIN_POS)
        
        self.write_position(pos)
        self.trigger_motion()

    def read_pos(self):
        """
        Read the gripper position from the feedback position register with ~0.0157s.

        Returns:
            float: The normalized gripper width in range [0,1], where 0 represents fully 
                closed and 1 represents fully open position. The value is calculated by 
                reading the high 8 bits of the feedback position register and normalizing 
                it between the closed and open feedback positions.

        Raises:
            AssertionError: If the read position is outside valid range (0 to MAX_FEEDBACK_POS) 
                or if the calculated normalized width is outside [0,1].
        
        """
        pos_high8 = self.instrument.read_long(self.FEEDBACK_POS_HIGH_8_REG)
        
        assert 0 <= pos_high8 <= self.MAX_FEEDBACK_POS or pos_high8 > 10000, \
            f"pos_high8: {pos_high8}, type of pos_high8: {type(pos_high8)}"

        if pos_high8 > 10000:
            pos_high8 = 0

        real_width = (pos_high8 - self.CLOSE_FEEDBACK_POS) / (self.OPEN_FEEDBACK_POS - self.CLOSE_FEEDBACK_POS)

        assert 0 <= real_width <= 1, \
            f"real_width: {real_width}, type of real_width: {type(real_width)}"
        
        return real_width

    def read_serial(self):
        return str(self.instrument.read_long(self.ID_REG))