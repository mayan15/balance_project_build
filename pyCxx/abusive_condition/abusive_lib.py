# -*- coding: utf-8 -*-

import ctypes 
import numpy as np
import pandas as pd

class Abusive_condition(object):
    def __init__(self, libname):
        self.clib = ctypes.CDLL(libname)

    def abusive_judge(self, voltage_pd,vol_ub,vol_lb,temperature_pd,temp_ub,temp_lb,current_pd,full_cap):
        # 指定函数返回类型和参数类型
        self.clib.countConsecutiveGreaterThan.restype = ctypes.c_int
        self.clib.countConsecutiveGreaterThan.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
        
        self.clib.countGreaterThan.restype = ctypes.c_int
        self.clib.countGreaterThan.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]

        self.clib.countLessThan.restype = ctypes.c_int
        self.clib.countLessThan.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]

        
        # 将 Pandas Series 转换为一维数组
        voltage = voltage_pd.values.flatten()
        temperature = temperature_pd.values.flatten()
        current = current_pd.values.flatten()

        # 将 Python 列表转换为 ctypes 数组
        voltage_type = ctypes.c_float * len(voltage)
        voltage_ctype = voltage_type(*voltage)
        voltage_size_ctype = ctypes.c_int(len(voltage))
        vol_ub_ctype = ctypes.c_float(vol_ub)
        vol_lb_ctype = ctypes.c_float(vol_lb)
        
        # print(voltage)
        # print(len(voltage))
        # print(vol_ub_ctype)
        over_charge_num = self.clib.countGreaterThan(voltage_ctype, voltage_size_ctype, vol_ub_ctype)
        over_discharge_num = self.clib.countLessThan(voltage_ctype, voltage_size_ctype, vol_lb_ctype)

        temperature_type = ctypes.c_float * len(temperature)
        temperature_ctype = temperature_type(*temperature)
        temperature_size_ctype = ctypes.c_int(len(temperature))
        temp_ub_ctype = ctypes.c_float(temp_ub)
        temp_lb_ctype = ctypes.c_float(temp_lb)

        high_temp_num = self.clib.countGreaterThan(temperature_ctype, temperature_size_ctype, temp_ub_ctype)
        low_temp_num = self.clib.countLessThan(temperature_ctype, temperature_size_ctype, temp_lb_ctype)

        current_type = ctypes.c_float * len(current)
        current_ctype = current_type(*current)
        current_size_ctype = ctypes.c_int(len(current))
        current_ub_ctype = ctypes.c_float(full_cap)
        high_rate_num = self.clib.countConsecutiveGreaterThan(current_ctype,current_size_ctype,current_ub_ctype)
        
        return over_charge_num,over_discharge_num,high_temp_num,low_temp_num,high_rate_num
