# Deployment Tips for RDT2

This document provides tips and best practices for deploying RDT2 in various environments.

* Perform proper calibration; you can use the replay script to check whether calibration is successful. 
  ```
    python scripts/replay_calibration.py --config <path_to_your_config_file>
  ``` 
* The fisheye camera has a limited field of view, so place objects in a clearly visible position. During inference, you can adjust placement by observing the image visualization interface.
* Use a soft pad on the table (e.g., thick foam), as the Zhixing gripper may cause the robot to trigger an emergency stop if it touches a hard surface due to excessive torque.
* Adjust the initial pose properly. We provide initial poses for FR3 and UR5e; for other robots, refer to the provided image to adjust the initial pose.
* If objects frequently slip from the Zhixing gripper, try applying anti-slip tape to the gripper.
* Separate pick and place actions into two prompts for the model, as our pretrained model does not currently support task composition.
