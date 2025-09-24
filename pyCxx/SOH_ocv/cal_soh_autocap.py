import os
import pandas as pd
import numpy as np
import time

from tool_utils.pvlog import Logger

'''
获取充电的起始和结束电压，寻找对应ocv，然后计算soh
'''
level = 'debug'
log = Logger('./logs/cal_soh_autocap.log', level=level)


class CalSOH:
    def __init__(self, charge_cutoff_voltage=2.5, discharge_cutoff_voltage=4.2):
        self.ocv_nmc_charge = [
            3.0991, 3.282, 3.3804, 3.4102, 3.4238, 3.4391, 3.4593, 3.4824, 3.5045, 3.5236,
            3.5415, 3.5577, 3.5698, 3.58, 3.5882, 3.5955, 3.6023, 3.6088, 3.6152, 3.6213,
            3.6276, 3.6342, 3.6411, 3.6485, 3.6566, 3.6657, 3.6766, 3.6916, 3.7133, 3.7348,
            3.7535, 3.7717, 3.7902, 3.8095, 3.8292, 3.8494, 3.87, 3.8904, 3.9113, 3.9323,
            3.9537, 3.9758, 3.998, 4.0219, 4.0453, 4.0692, 4.0948, 4.1205, 4.1472, 4.1783,
            4.2094]
        self.soc_nmc_charge = [
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
            34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0,
            66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0,
            98.0, 100.0]
        
        self.ocv_nmc_discharge = [
            2.7305, 2.9186, 3.0991, 3.2803, 3.3697, 3.4014, 3.4151, 3.4294, 3.4475, 3.4671,
            3.4863, 3.5038, 3.5194, 3.5344, 3.549, 3.5625, 3.5741, 3.5836,
            3.5915, 3.5986, 3.605, 3.6112, 3.6175, 3.6241, 3.631, 3.6381,
            3.6461, 3.6552, 3.6664, 3.6812, 3.7007, 3.7227, 3.7432, 3.7624,
            3.7816, 3.8009, 3.8208, 3.8408, 3.8613, 3.8819, 3.9026, 3.9235,
            3.9449, 3.9668, 3.9894, 4.0127, 4.0366, 4.0617, 4.087, 4.1131,
            4.141, 4.1711, 4.2094, 4.2505, 4.2952, 4.3423, 4.3918, 4.4437]
        self.soc_nmc_discharge = [
            -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
            34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0,
            66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0,
            98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0]

        self.ocv_lfp_charge = [
            2.353, 2.93, 3.084, 3.175, 3.219, 3.222, 3.224, 3.229, 3.24, 3.254, 
            3.266, 3.278, 3.287, 3.297, 3.3, 3.303, 3.304, 3.305, 3.305, 3.306, 
            3.306, 3.307, 3.307, 3.308, 3.308, 3.309, 3.31, 3.311, 3.311, 3.313, 
            3.315, 3.32, 3.34, 3.346, 3.346, 3.346, 3.346, 3.346, 3.346, 3.347, 
            3.347, 3.347, 3.347, 3.347, 3.347, 3.347, 3.347, 3.346, 3.345, 3.345, 3.369]
        self.soc_lfp_charge = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 
            20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 
            40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 
            60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 
            80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]
        
        self.ocv_lfp_discharge = [
            2.292, 2.935, 3.076, 3.162, 3.195, 3.199, 3.201, 3.202, 3.213, 3.228, 
            3.24, 3.251, 3.257, 3.264, 3.271, 3.277, 3.281, 3.283, 3.285, 3.285, 
            3.286, 3.286, 3.287, 3.287, 3.287, 3.288, 3.289, 3.289, 3.29, 3.291, 
            3.294, 3.303, 3.319, 3.326, 3.327, 3.327, 3.328, 3.328, 3.328, 3.328, 
            3.328, 3.328, 3.328, 3.329, 3.329, 3.329, 3.33, 3.33, 3.33, 3.332, 3.369]
        self.soc_lfp_discharge = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 
            20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 
            40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 
            60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 
            80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]

    def scale_soc_ocv(self, cell_type, charge_cutoff_voltage_new, discharge_cutoff_voltage_new):
        if cell_type == 'nmc':
            # SOC缩放（原代码是OCV缩放）
            soc_nmc_copy = self.soc_nmc.copy()  # 获取原始SOC数据
            ocv_nmc_copy = self.ocv_nmc.copy()  # 获取原始OCV数据

            # 获取截止电压对应的原始SOC值
            charge_cutoff_voltage_original = 4.25
            discharge_cutoff_voltage_original = 2.5
            diff_cutoff_voltage_original = charge_cutoff_voltage_original - discharge_cutoff_voltage_original

            # 获取新的截止电压在原始ocv曲线上对应的soc值,%
            soc_original_curve_charge_cutoff_voltage_new = (1 - (charge_cutoff_voltage_original - charge_cutoff_voltage_new) * 0.522) * 100
            soc_original_curve_discharge_cutoff_voltage_new = (discharge_cutoff_voltage_new - discharge_cutoff_voltage_original) * 0.0343 * 100

            # 定义新的SOC范围（通常充电截止对应100%，放电截止对应0%）
            new_charge_soc = 100.0
            new_discharge_soc = 0.0

            # 计算SOC的映射关系：soc_new = a*soc_original + b
            # 确保映射后充电截止电压对应100% SOC，放电截止电压对应0% SOC
            a = (new_charge_soc - new_discharge_soc) / (soc_original_curve_charge_cutoff_voltage_new - soc_original_curve_discharge_cutoff_voltage_new)
            b = new_charge_soc - a * soc_original_curve_charge_cutoff_voltage_new

            # 计算新的SOC值
            self.soc_nmc = [a * soc + b for soc in soc_nmc_copy]  # 列表推导式逐个计算

            # OCV值保持不变，与新的SOC值对应
            self.ocv_nmc = ocv_nmc_copy

    def get_lfp_soc_from_ocv(self, vol, is_charge=False):
        if is_charge:
            ocv_lfp = self.ocv_lfp_charge.copy()
            soc_lfp = self.soc_lfp_charge.copy()
        else:
            ocv_lfp = self.ocv_lfp_discharge.copy()
            soc_lfp = self.soc_lfp_discharge.copy()

        iii = 0
        # ocv
        for index, item in enumerate(ocv_lfp):
            if vol < item:
                iii = index
                break
        if iii == 0 and vol > ocv_lfp[-1]:
            iii = len(ocv_lfp) - 1

        cal_soc = None
        if iii > 0:
            soc_start = soc_lfp[iii - 1]
            numerator = 2 * (vol - ocv_lfp[iii - 1])
            denominator = (ocv_lfp[iii] - ocv_lfp[iii - 1])
            cal_soc = soc_start + numerator / denominator
        return cal_soc

    def get_nmc_soc_from_ocv(self, vol, is_charge=False):
        if is_charge:
            ocv_nmc = self.ocv_nmc_charge.copy()
            soc_nmc = self.soc_nmc_charge.copy()
        else:
            ocv_nmc = self.ocv_nmc_discharge.copy()
            soc_nmc = self.soc_nmc_discharge.copy()

        iii = 0
        for index, item in enumerate(ocv_nmc):
            if vol < item:
                iii = index
                break

        if iii == 0 and vol > ocv_nmc[-1]:
            iii = len(ocv_nmc) - 1

        cal_soc = 0
        if iii > 0:
            soc_start = soc_nmc[iii - 1]
            numerator = 2 * (vol - ocv_nmc[iii - 1])
            denominator = (ocv_nmc[iii] - ocv_nmc[iii - 1])
            cal_soc = soc_start + numerator / denominator
        return cal_soc

import numpy as np
# 假设 CalSOH 和 log 已在外部导入

class SOHCalculator:
    """
    电芯容量和SOH计算类
    """
    def __init__(self, charge_cutoff_voltage=3.8, discharge_cutoff_voltage=2.0):
        self.cal = CalSOH()
        self.charge_cutoff_voltage = charge_cutoff_voltage
        self.discharge_cutoff_voltage = discharge_cutoff_voltage

        # 异常记录
        self.error_records = ''

    def calculate_cell_capacity(self, dcap, start_vol, end_vol, cell_type, cell_id):
        """根据起始/结束电压计算单体电芯容量"""
        try:
            # # 20250916 my添加 手动制造异常结果
            # if cell_id == '1':
            #     return 10
            if cell_type.lower() == 'lfp':
                if start_vol > end_vol:
                    start_soc = self.cal.get_lfp_soc_from_ocv(start_vol, is_charge=False)
                    end_soc = self.cal.get_lfp_soc_from_ocv(end_vol, is_charge=False)
                else:
                    start_soc = self.cal.get_lfp_soc_from_ocv(start_vol, is_charge=True)
                    end_soc = self.cal.get_lfp_soc_from_ocv(end_vol, is_charge=True)
            elif cell_type.lower() in ('nmc', 'ncm'):
                if start_vol > end_vol:
                    start_soc = self.cal.get_nmc_soc_from_ocv(start_vol, is_charge=False)
                    end_soc = self.cal.get_nmc_soc_from_ocv(end_vol, is_charge=False)
                else:
                    start_soc = self.cal.get_nmc_soc_from_ocv(start_vol, is_charge=True)
                    end_soc = self.cal.get_nmc_soc_from_ocv(end_vol, is_charge=True)
            else:
                self.error_records += f'{cell_id}号电芯{cell_type}不是有效的电池类型，无法计算容量。'
                log.logger.error(f'计算容量出错: {cell_type}不是有效类型')
                return None

            if start_soc is None or end_soc is None:
                self.error_records += f'{cell_id}号电芯电压超出范围，无法计算。'
                return None

            soc_diff = abs(end_soc - start_soc) / 100
            if soc_diff == 0:
                self.error_records += f'{cell_id}号电芯SOC变化为0，无法计算容量。'
                return None
            elif dcap == 0:
                self.error_records += f'{cell_id}号电芯容量变化为0，无法计算容量。'
                return None

            return dcap / soc_diff
        except Exception as e:
            self.error_records += f'{cell_id}号电芯计算出错: {str(e)}'
            raise RuntimeError(f'calculate_cell_capacity函数出错,{e}') from e

    def cal_soh_scale_and_offset(self, all_cell_no_list, all_capacity_list, 
                                 available_cell_no_list, available_capacity_list, 
                                 standard_capacity, is_scale=True):
        """根据容量计算SOH并进行修正"""
        try:
            if len(available_cell_no_list) > 0:
                capacity_array = np.array(available_capacity_list)
                available_soh_array = capacity_array / standard_capacity * 100

                if is_scale:
                    max_soh = np.max(available_soh_array)
                    min_soh = np.min(available_soh_array)
                    median_soh = np.median(available_soh_array)

                    if max_soh >= 100 and min_soh < 100 and (max_soh - median_soh) <= 5:
                        values_less_100 = available_soh_array[available_soh_array < 100]
                        max_less_than100 = max(values_less_100) if len(values_less_100) > 0 else 100
                        shift = max_soh - max_less_than100
                        available_soh_array = available_soh_array - shift
                    elif min_soh > 100 and (max_soh - median_soh) <= 5:
                        target_diff = 2
                        current_diff = max_soh - median_soh
                        scale_factor = target_diff / current_diff if current_diff != 0 else 1
                        scaled_soh = (available_soh_array - median_soh) * scale_factor + median_soh
                        new_median = np.median(scaled_soh)
                        shift = new_median - 95
                        available_soh_array = scaled_soh - shift
                    elif max_soh < 100 and (max_soh - median_soh) > 5:
                        target_range = 5
                        current_range = max_soh - min_soh
                        scale_factor = target_range / current_range if current_range != 0 else 1
                        available_soh_array = (available_soh_array - median_soh) * scale_factor + median_soh
                    elif (max_soh - median_soh) > 5:
                        shift = max_soh - max(min(100, median_soh), 90)
                        shifted_soh = available_soh_array - shift
                        new_max = max(shifted_soh)
                        new_min = min(shifted_soh)
                        new_median = np.median(shifted_soh)
                        new_range = new_max - new_min
                        if new_range > 5:
                            target_range = 5
                            scale_factor = target_range / new_range if new_range != 0 else 1
                            available_soh_array = (shifted_soh - new_median) * scale_factor + new_median
                        else:
                            available_soh_array = shifted_soh
                    elif median_soh < 60:
                        shift = 85 - median_soh
                        available_soh_array = available_soh_array + shift

            else:
                available_soh_array = np.array([])

            all_soh_list = []
            for cell_no in all_cell_no_list:
                if cell_no in available_cell_no_list:
                    index = available_cell_no_list.index(cell_no)
                    all_soh_list.append(available_soh_array[index])
                else:
                    all_soh_list.append(None)

            return np.array(all_soh_list), available_soh_array
        except Exception as e:
            self.error_records += f'SOH映射过程出错。'
            raise RuntimeError(f'cal_soh_scale_and_offset函数出错,{e}') from e

    def get_soh_results(self, autocap_rlt_res, data_clean_rlt):
        """获取每个电芯的SOH结果"""
        try:
            cell_type = data_clean_rlt['out']['battery_type'][0]
            standard_capacity = data_clean_rlt['out']['battery_capacity'][0]
            cell_no_list = autocap_rlt_res['cell_no_list']
            autocap_data = autocap_rlt_res['autocap_data']

            all_cell_no_list, available_cell_no_list, invalid_cell_no_list = [], [], []
            all_capacity_list, available_capacity_list = [], []

            for cell_no in cell_no_list:
                cell_id = str(cell_no)
                cell_data = autocap_data[cell_id]
                dcap, start_vol, end_vol = cell_data['dcap'], cell_data['start_vol'], cell_data['end_vol']
                capacity = self.calculate_cell_capacity(dcap, start_vol, end_vol, cell_type, cell_id)

                all_cell_no_list.append(cell_no)
                all_capacity_list.append(capacity)
                if capacity is not None:
                    available_cell_no_list.append(cell_no)
                    available_capacity_list.append(capacity)
                else:
                    invalid_cell_no_list.append(cell_no)

            all_soh_array, available_soh_array = self.cal_soh_scale_and_offset(
                all_cell_no_list, all_capacity_list, available_cell_no_list, available_capacity_list, standard_capacity
            )
            
            return all_cell_no_list, all_soh_array, available_cell_no_list, available_soh_array, invalid_cell_no_list
        except Exception as e:
            self.error_records += f'SOH结果获取过程出错。'
            raise RuntimeError(f'get_soh_results函数出错,{e}') from e

    @staticmethod
    def add_out_dir_info(rlt_res, key, value, confidence, explanation):
        """向结果字典中添加字段"""
        rlt_res['out'][key] = [value, confidence, explanation]

    def get_output_json(self, rlt_res, all_cell_no_list, all_soh_array, 
                        available_cell_no_list, available_soh_array, 
                        invalid_cell_no_list):
        """生成输出json"""
        try:
            # 结果初始化
            max_soh = None
            min_soh = None

            cell_no_max_soh = None
            cell_no_min_soh = None

            soh_abnormal_cell_no_str = ''
            all_cell_no_list_str = ''
            all_soh_array_str = ''

            all_cell_no_list_str = ','.join(str(i) for i in all_cell_no_list)
            invalid_cell_no_list_str = ','.join(str(i) for i in invalid_cell_no_list)
            all_soh_array_str = ','.join(str(round(i, 1)) if i is not None else 'None' for i in all_soh_array)

            if len(available_cell_no_list) > 0:
                max_soh = round(np.max(available_soh_array), 2)
                min_soh = round(np.min(available_soh_array), 2)
                # 数字，整数形式的电芯号，找到available_soh_array最大值对应位置available_cell_no_list值
                cell_no_max_soh = available_cell_no_list[np.argmax(available_soh_array)]
                cell_no_min_soh = available_cell_no_list[np.argmin(available_soh_array)]

                # soh异常电芯号，soh低于80且与中值差值大于5的电芯
                # 同时记录除异常电芯外最小SOH值
                soh_abnormal_low_cell_no_list = []
                soh_cell_no_lower80_list = []
                soh_cell_no_lower60_list = []
                normal_soh_values = []  # 存储正常电芯的SOH值
                median_soh = np.median(available_soh_array)

                for ii in range(len(available_soh_array)):
                    current_soh = available_soh_array[ii]
                    if current_soh < 80 and (median_soh - current_soh) > 5:
                        soh_abnormal_low_cell_no_list.append(ii + 1)  # 记录异常电芯号（+1表示从1开始编号）
                    elif current_soh < 60:
                        soh_cell_no_lower60_list.append(ii + 1)  # 记录SOH低于60的电芯号
                    elif current_soh < 80:
                        soh_cell_no_lower80_list.append(ii + 1)  # 记录SOH低于80的电芯号
                    else:
                        normal_soh_values.append(current_soh)  # 收集正常电芯的SOH值

                # 计算除异常电芯外的最小SOH值
                if normal_soh_values:  # 确保存在正常电芯
                    min_normal_soh = min(normal_soh_values)
                else:
                    min_normal_soh = None  # 或根据需求设置默认值，如0

                # soh异常电芯号列表转为字符串
                soh_abnormal_cell_no_str = ','.join(str(i) for i in soh_abnormal_low_cell_no_list) if len(soh_abnormal_low_cell_no_list) > 0 else '/'
                # 容量测试电芯号转为字符串
                available_cell_no_list_str = ','.join(str(i) for i in available_cell_no_list) if len(available_cell_no_list) > 0 else '/'
                
                # 容量测试SOH值转为字符串
                available_soh_list_str = ','.join(str(round(i,1)) for i in available_soh_array) if len(available_soh_array) > 0 else '/'
                
                # 建议及结果说明字段
                summary = '/'
                advice = '/'
                result_desc = '/'
                if len(available_cell_no_list) != len(all_cell_no_list):
                    summary = f'进行容量测试的{len(all_cell_no_list)}节电芯中，共有效计算{len(available_cell_no_list)}节电芯SOH，第{cell_no_max_soh}节电芯SOH最大为{max_soh:.2f}%，第{cell_no_min_soh}节电芯SOH最小为{min_soh:.2f}%。'
                    result_desc = f'本次进行SOH测试的电芯号依次为{all_cell_no_list_str}，其中部分电芯无有效计算结果，电芯号为{invalid_cell_no_list_str}。具有有效计算结果的电芯号为{available_cell_no_list_str}，对应SOH值（单位%）分别为{available_soh_list_str}。'
                    if soh_abnormal_low_cell_no_list:
                        advice = f'部分电芯SOH存在异常，建议更换异常电芯。'   # ，维护后整组SOH最高可提升至{min_normal_soh:.2f}%
                    else:
                        advice = f'暂无建议，具有有效计算结果的电芯全部SOH正常。'
                else:
                    summary = f'进行容量测试的{len(all_cell_no_list)}节电芯中，第{cell_no_max_soh}节电芯SOH最大为{max_soh:.2f}%，第{cell_no_min_soh}节电芯SOH最小为{min_soh:.2f}%。'
                    result_desc = f'本次进行SOH测试的电芯号依次为{all_cell_no_list_str}，全部具有有效计算结果，对应SOH值（单位%）分别为{available_soh_list_str}。'
                    if soh_abnormal_low_cell_no_list:
                        advice = f'部分电芯SOH存在异常，建议更换异常电芯。'  # ，维护后整组SOH最高可提升至{min_normal_soh:.2f}%
                    elif len(soh_cell_no_lower60_list) > len(all_cell_no_list) * 0.5:
                        self.error_records += f'SOH值整体异常偏低。'
                        self.invalid_result_handle(rlt_res)
                        # return rlt_res  # 异常结果处理
                    elif len(soh_cell_no_lower80_list) > len(all_cell_no_list) * 0.5:
                        advice = f'本次进行SOH测试的全部电芯中，一半以上电芯SOH低于80%。建议人工复验，确认则建议更换电池包。'
                    else:
                        advice = f'暂无建议，容量测试电芯SOH全部正常。'
            elif len(all_cell_no_list) > 0:
                summary = f'进行容量测试的{len(all_cell_no_list)}节电芯中，全部无有效计算结果，建议人工排查或联系售后人员。'
                result_desc = f'本次进行SOH测试的电芯号依次为{all_cell_no_list_str}，全部无有效计算结果。'
                advice = '暂无维修建议，本次SOH测试无有效计算结果。'
            else:
                summary = '本次没有进行SOH测试的电芯，或输入的电芯序号与实际测试通道号匹配有误，无有效计算结果。'
                result_desc = '本次没有进行SOH测试的电芯，或输入的电芯序号与实际测试通道号匹配有误，无有效计算结果。'
                advice = '暂无维修建议。'

            self.add_out_dir_info(rlt_res, '容量测试电芯号', all_cell_no_list_str, '', '')
            self.add_out_dir_info(rlt_res, '容量测试SOH值', all_soh_array_str, '', '')
            self.add_out_dir_info(rlt_res, 'SOH最高值', max_soh, '', '')
            self.add_out_dir_info(rlt_res, 'SOH最低值', min_soh, '', '')
            self.add_out_dir_info(rlt_res, 'SOH最高值电芯号', cell_no_max_soh, '', '')
            self.add_out_dir_info(rlt_res, 'SOH最低值电芯号', cell_no_min_soh, '', '')
            self.add_out_dir_info(rlt_res, 'SOH值异常电芯号', soh_abnormal_cell_no_str, '', '')
            self.add_out_dir_info(rlt_res, '建议', advice, '', '')
            self.add_out_dir_info(rlt_res, '说明', summary, '', '')
            # self.add_out_dir_info(rlt_res, '容量测试详细说明', result_desc, '', '')
            self.add_out_dir_info(rlt_res, 'SOH计算异常说明', self.error_records, '', '')

            return rlt_res
        except Exception as e:
            self.error_records += f'获取json结果过程出错。'
            raise RuntimeError(f'get_output_json函数出错,{e}') from e
        
    def invalid_result_handle(self, rlt_res):
        """处理无效结果"""
        self.add_out_dir_info(rlt_res, '容量测试电芯号', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '容量测试SOH值', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, 'SOH最高值', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, 'SOH最低值', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, 'SOH最高值电芯号', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, 'SOH最低值电芯号', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, 'SOH值异常电芯号', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '建议', '暂无建议。', '', '')
        self.add_out_dir_info(rlt_res, '说明', '容量测试没有有效数据或计算结果。', '', '')
        self.add_out_dir_info(rlt_res, 'SOH计算异常说明', self.error_records, '', '')

def run(autocap_rlt_res, data_clean_rlt):
    """
    对外接口：调用 SOHCalculator 返回结果
    """
    rlt_res = {
        "code_id": 2,
        "describe": "soh",
        "out": {},
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }
    try:
        st = time.time()

        soh_calc = SOHCalculator()
        all_cell_no_list, all_soh_array, available_cell_no_list, available_soh_array, invalid_cell_no_list = soh_calc.get_soh_results(autocap_rlt_res, data_clean_rlt)
        soh_info = soh_calc.get_output_json(rlt_res, all_cell_no_list, all_soh_array, available_cell_no_list, available_soh_array, invalid_cell_no_list)
        
        log.logger.debug(f"soh calculate time: {round(time.time()-st,5)} seconds")
        return soh_info
    except Exception as e:
        rlt_res['ErrorCode'][0] = 2001
        soh_calc.invalid_result_handle(rlt_res)
        log.logger.error(f'获取SOH结果出错: {e}')
        return rlt_res
