import copy
from dataclasses import field, dataclass
from typing import Callable, Optional, Any, Dict
import multiprocessing as mp
import traceback

import sys
import os
import time
from ctypes import *

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from deploy.umi.real_world.camera.cam import (CamController,
                                              CamControllerConfig)

try:
    sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
    from MvCameraControl_class import *

except ImportError:
    print("Failed to import MvCameraControl_class. Ensure the MVS SDK is installed and the path is correct.")


@dataclass
class MVSCamControllerConfig(CamControllerConfig):
    serial: str = ""

    width: int = 480
    height: int = 480

    transformed_width: int = 480
    transformed_height: int = 480

    crop_func: Callable = lambda img: img[61:1057, 224:1235, :]

    def validate(self):
        super().validate()
        assert isinstance(self.serial, str) and len(self.serial) > 0, f"Invalid camera serial: {self.serial}"

def get_all_mvs_dev_serial():
    """
    获取所有MVS设备的序列号。
    """

    deviceList = MV_CC_DEVICE_INFO_LIST()
    layerType = (MV_GIGE_DEVICE
                 | MV_USB_DEVICE
                 | MV_GENTL_CAMERALINK_DEVICE
                 | MV_GENTL_CXP_DEVICE
                 | MV_GENTL_XOF_DEVICE)

    ret = MvCamera.MV_CC_EnumDevices(layerType, deviceList)
    if ret != 0:
        print("Enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    serial_numbers = []

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        strSerialNumber = ""

        if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber += chr(per)
            serial_numbers.append(strSerialNumber)

    return serial_numbers

class MVSCamController(CamController):
    config: MVSCamControllerConfig

    def __init__(self, config: MVSCamControllerConfig):
        super().__init__(config)

    ################## cls methods ##################
    def initialize_sdk(self):
        """
        初始化SDK。
        """
        MvCamera.MV_CC_Initialize()
        sdk_version = MvCamera.MV_CC_GetSDKVersion()
        print("SDK Version: [0x%x]" % sdk_version)

    def enumerate_devices(self):
        """
        枚举可用设备，并将结果存储到 self.deviceList。
        """
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        layerType = (MV_GIGE_DEVICE
                     | MV_USB_DEVICE
                     | MV_GENTL_CAMERALINK_DEVICE
                     | MV_GENTL_CXP_DEVICE
                     | MV_GENTL_XOF_DEVICE)

        ret = MvCamera.MV_CC_EnumDevices(layerType, self.deviceList)
        if ret != 0:
            print("Enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if self.deviceList.nDeviceNum == 0:
            print("No device found!")
            sys.exit()

        print("Find %d devices!" % self.deviceList.nDeviceNum)

        self.DevSerialNumbers = []

        for i in range(0, self.deviceList.nDeviceNum):
            strSerialNumber = ""

            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            # 根据不同接口类型，打印相关信息
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))

            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                print("\nCML device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
                print("\nXoF device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                print("\nCXP device: [%d]" % i)
                strModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName:
                    if per == 0:
                        break
                    strModelName += chr(per)
                print("device model name: %s" % strModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber += chr(per)
                print("user serial number: %s" % strSerialNumber)

            self.DevSerialNumbers.append(strSerialNumber)

    def connect_device(self, index):
        """
        根据索引连接指定设备，并完成打开设备、设置包大小、关闭触发模式等操作。
        """
        # 创建相机实例
        self.cam = MvCamera()

        # 获取设备信息
        self.stDeviceList = cast(self.deviceList.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents

        # 创建句柄
        ret = self.cam.MV_CC_CreateHandle(self.stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            sys.exit()

        # 若为GigE接口，则设置最佳包大小
        if self.stDeviceList.nTLayerType == MV_GIGE_DEVICE or self.stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # 设置触发模式为off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # 获取数据包大小
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()

        self.nPayloadSize = stParam.nCurValue
        self.data_buf = (c_ubyte * self.nPayloadSize)()

    ################## abstract methods ##################
    def _process_commands(self):
        super()._process_commands()

    def _initialize(self):
        self.initialize_sdk()
        self.enumerate_devices()

        self.connect_device(self.DevSerialNumbers.index(self.config.serial))

        ret = self.cam.MV_CC_StartGrabbing()
        assert ret == 0, f"Start grabbing fail! ret[0x{ret:x}]"

        # ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateControlEnable", 1)
        # ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 30.0)

        # MV_EXPOSURE_AUTO_MODE_OFF, MV_EXPOSURE_AUTO_MODE_CONTINUOUS
        # ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_CONTINUOUS)
        # ret = self.cam.MV_CC_SetIntValue("AutoExposureTimeLowerLimit", 15)
        # ret = self.cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 32000)
        
        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        ret = self.cam.MV_CC_SetIntValue("ExposureTime", 20000)

        # MV_GAIN_MODE_OFF, MV_GAIN_MODE_CONTINUOUS
        # ret = self.cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
        # ret = self.cam.MV_CC_SetFloatValue("AutoGainLowerLimit", 0.0)
        # ret = self.cam.MV_CC_SetFloatValue("AutoGainUpperLimit", 16.9807)
        ret = self.cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_CONTINUOUS)

        # MV_BALANCEWHITE_AUTO_OFF, MV_BALANCEWHITE_AUTO_CONTINUOUS
        ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)

        ret = self.cam.MV_CC_SetBoolValue("BlackLevelEnable", 1)
        ret = self.cam.MV_CC_SetIntValue("BlackLevel", 240)

        ret = self.cam.MV_CC_SetIntValue("Brightness", 30)

        self.stOutFrame = MV_FRAME_OUT()
        memset(byref(self.stOutFrame), 0, sizeof(self.stOutFrame))

        super()._initialize()

    def _update(self):
        ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 1000)
        assert ret == 0, f"Get image buffer fail! ret[0x{ret:x}]"

        self.last_timestamp = time.time()

        nRGBSize = self.stOutFrame.stFrameInfo.nWidth * self.stOutFrame.stFrameInfo.nHeight * 3

        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
        stConvertParam.nWidth = self.stOutFrame.stFrameInfo.nWidth
        stConvertParam.nHeight = self.stOutFrame.stFrameInfo.nHeight
        stConvertParam.pSrcData = self.stOutFrame.pBufAddr
        stConvertParam.nSrcDataLen = self.stOutFrame.stFrameInfo.nFrameLen
        stConvertParam.enSrcPixelType = self.stOutFrame.stFrameInfo.enPixelType
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
        stConvertParam.nDstBufferSize = nRGBSize

        ret = self.cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
        if ret != 0:
            print ("convert pixel fail! ret[0x%x]" % ret)
            sys.exit()

        frame_data_ptr = cast(stConvertParam.pDstBuffer, POINTER(c_ubyte))
        frame_data = np.ctypeslib.as_array(frame_data_ptr, shape=(nRGBSize,))
        self.last_img = frame_data.reshape(self.stOutFrame.stFrameInfo.nHeight, self.stOutFrame.stFrameInfo.nWidth, 3)

        self.last_img = self.config.crop_func(self.last_img)
        self.last_img = cv2.resize(self.last_img, (480, 480), interpolation=cv2.INTER_LINEAR)
        self.last_img = cv2.cvtColor(self.last_img, cv2.COLOR_BGR2RGB)

        self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)

        super()._update()

    def _close(self):
        # TODO
        super()._close()

    def reset(self):
        super().reset()
