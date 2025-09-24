import sys
sys.path.append(".")

from deploy.umi.real_world.camera.mvs_cam import get_all_mvs_dev_serial

if __name__ == "__main__":
    serials = get_all_mvs_dev_serial()
    print(serials)