import sys
sys.path.append(".")
import os

from deploy.umi.real_world.zhixing_driver import ZhixingDriver

def get_all_zhixing_serials():
    """
    Args:
        lr_mapping: left | right: serial
    """
    # check all devs are /dev/ttyUSB*
    devs = os.listdir('/dev')
    devs = [dev for dev in devs if dev.startswith('ttyUSB')]
    dev_ids = [int(dev.split('ttyUSB')[-1]) for dev in devs]

    serials = []
    for dev_id in dev_ids:
        try:
            with ZhixingDriver(serial_dev=f"/dev/ttyUSB{dev_id}") as zx:
                zx_serial = zx.read_serial()
                serials.append(zx_serial)
        except Exception as e:
            # print(f"Error reading serial from /dev/ttyUSB{dev_id}: {e}")
            continue

    return serials

if __name__ == "__main__":
    serials = get_all_zhixing_serials()
    print(serials)