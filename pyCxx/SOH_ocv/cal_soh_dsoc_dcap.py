import os
import pandas as pd
import numpy as np
from tool_utils.pvlog import Logger

'''
获取充电的起始和结束电压，寻找对应ocv，然后计算soh
'''
level = 'info'
log = Logger('./logs/run_alg.log', level=level)


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
        cal_soc = 0
        if iii > 0:
            soc_start = soc_nmc[iii - 1]
            numerator = 2 * (vol - ocv_nmc[iii - 1])
            denominator = (ocv_nmc[iii] - ocv_nmc[iii - 1])
            cal_soc = soc_start + numerator / denominator
        return cal_soc

def calculate_cell_capacities_by_voltages(dcap, start_vol, end_vol, cell_type, cell_id, charge_cutoff_voltage=3.8, discharge_cutoff_voltage=2.0):
    '''
    计算指定CSV文件中各电芯的容量，按电芯序号循环匹配对应列

    参数:
        file_paths: 要处理的CSV文件路径列表
        cal: 包含get_lfp_soc_for_ocv方法的对象

    返回:
        各电芯容量的字典
    '''
    try:
        cal = CalSOH()
        error_record = ''

        # 根据截至电压，缩放soc列表
        # cal.scale_soc_ocv(cell_type, charge_cutoff_voltage, discharge_cutoff_voltage)

        # 计算对应的SOC，OCV表根据截止电压缩放
        if cell_type == 'lfp' or cell_type == 'LFP':
            if start_vol > end_vol:
                start_soc = cal.get_lfp_soc_from_ocv(start_vol, is_charge=False)
                end_soc = cal.get_lfp_soc_from_ocv(end_vol, is_charge=False)                
                charge_or_discharge = '放电'
            else:
                start_soc = cal.get_lfp_soc_from_ocv(start_vol, is_charge=True)
                end_soc = cal.get_lfp_soc_from_ocv(end_vol, is_charge=True)  
                charge_or_discharge = '充电'
            
        elif cell_type == 'nmc' or cell_type == 'ncm' or cell_type == 'NMC' or cell_type == 'NCM':
            if start_vol > end_vol:
                start_soc = cal.get_nmc_soc_from_ocv(start_vol, is_charge=False)
                end_soc = cal.get_nmc_soc_from_ocv(end_vol, is_charge=False)
                charge_or_discharge = '放电'
            else:
                start_soc = cal.get_nmc_soc_from_ocv(start_vol, is_charge=True)
                end_soc = cal.get_nmc_soc_from_ocv(end_vol, is_charge=True)
                charge_or_discharge = '充电'
        else:
            error_record += f'{cell_id}号电芯{cell_type}不是有效的电池类型，无法计算容量。'
            log.logger.error(f'计算容量calculate_cell_capacities_by_voltages函数出错，错误信息：{cell_type}不是有效的电池类型。')
            return None, error_record

        # 电压超出搜索范围
        if start_soc == None or end_soc == None:
            error_record += f'{cell_id}号电芯数据电压超过有效范围，无法计算。'
            log.logger.info(f'{cell_id}计算SOC值出错，起始电压{start_vol}，结束电压{end_vol}。')
            return None, error_record

        # 计算SOC差值（转换为0-1范围）
        soc_diff = abs(end_soc - start_soc) / 100
        # 测试异常用
        # if cell_id == '1':
        #     soc_diff = 0

        # soc变化为0或者容量变化为0
        if soc_diff == 0:
            error_record += f'{cell_id}号电芯SOC变化为0，无法计算容量。'
            log.logger.info(f'{cell_id}计算SOC差值为0，无法计算容量，起始SOC{start_soc}，结束SOC{end_soc}，起始电压{start_vol}，结束电压{end_vol}。')
            return None, error_record
        elif dcap == 0:
            error_record += f'{cell_id}号电芯容量变化为0，无法计算容量。'
            log.logger.info(f'{cell_id}电池容量变化为0，无法计算容量，起始电压{start_vol}，结束电压{end_vol}。')
            return None, error_record

        # 计算电芯容量
        cell_capacity = dcap / soc_diff
        # print(f'{charge_or_discharge}{dcap:.1f}Ah，计算容量结果:{cell_capacity:.2f}Ah    初始soc{start_soc:.2f}%  结束soc{end_soc:.2f}%')

        # 测试异常用
        # if cell_id == '1':
        #     return None

        return cell_capacity, error_record
    except Exception as e:
        log.logger.error(f'计算容量calculate_cell_capacities_by_voltages函数出错，错误信息：{e}')

def cal_soh_scale_and_offset(all_cell_no_list, all_capacity_list, available_cell_no_list, available_capacity_list, standard_capacity, is_scale=True):
    '''
    根据容量计算soh，并且确保soh值在合理范围内
    '''
    try:
        if len(available_cell_no_list) > 0:
            # 计算原始soh
            capacity_array = np.array(available_capacity_list)
            available_soh_array = capacity_array / standard_capacity * 100

            # 手动改值，测试用
            # soh_array = np.array([99]*len(soh_array))
            # soh_array[0] = 98
            # soh_array += 5

            # 存在异常值，可能是ocv与预设曲线存在差异，则进行映射
            # 最大soh异常，最小soh在合理范围内，且差值不大保证计算结果稳定，则进行偏移操作
            if is_scale:
                # 计算统计值
                max_soh = np.max(available_soh_array)
                min_soh = np.min(available_soh_array)
                median_soh = np.median(available_soh_array)

                if max_soh >= 100 and min_soh < 100 and (max_soh - median_soh) <= 5:
                    # 寻找第一个小于100的值（修正为寻找最大值中小于100的值）
                    values_less_100 = available_soh_array[available_soh_array < 100]
                    max_less_than100 = max(values_less_100) if len(values_less_100) > 0 else 100

                    # 计算位移为最大soh减去max_less_than100
                    shift = max_soh - max_less_than100

                    # 计算平移后的soh
                    available_soh_array = available_soh_array - shift
                
                elif min_soh > 100 and (max_soh - median_soh) <= 5:
                    # 缩放使得中值与最大SOH差值小于2
                    target_diff = 2
                    current_diff = max_soh - median_soh
                    scale_factor = target_diff / current_diff if current_diff != 0 else 1

                    # 计算新的中值，计算新中值与95之间位移差
                    scaled_soh = (available_soh_array - median_soh) * scale_factor + median_soh
                    new_median = np.median(scaled_soh)
                    shift = new_median - 95

                    # 计算平移后SOH
                    available_soh_array = scaled_soh - shift
                
                elif max_soh < 100 and (max_soh - median_soh) > 5:
                    # 缩放使得最小SOH值与最大SOH差值小于5，中值不变，其他增大或者缩小
                    target_range = 5
                    current_range = max_soh - min_soh
                    scale_factor = target_range / current_range if current_range != 0 else 1

                    # 计算缩放后SOH
                    available_soh_array = (available_soh_array - median_soh) * scale_factor + median_soh
                
                elif (max_soh - median_soh) > 5:
                    # 平移使得最大SOH为100或中值
                    shift = max_soh - max(min(100, median_soh), 90)
                    shifted_soh = available_soh_array - shift
                    
                    # 重新计算统计值
                    new_max = max(shifted_soh)
                    new_min = min(shifted_soh)
                    new_median = np.median(shifted_soh)
                    new_range = new_max - new_min
                    
                    # 缩放使得最小SOH值与最大SOH差值小于5，中值不变，其他增大或者缩小
                    if new_range > 5:
                        target_range = 5
                        scale_factor = target_range / new_range if new_range != 0 else 1
                        
                        # 计算缩放后SOH
                        available_soh_array = (shifted_soh - new_median) * scale_factor + new_median
                    else:
                        available_soh_array = shifted_soh
        else:
            available_soh_array = np.array([])

        # 对all_cell_no_list列表进行循环，如果在available_cell_no_list中有相同的，则找到available_soh_array对应值，否则为None
        all_soh_list = []
        for cell_no in all_cell_no_list:
            if cell_no in available_cell_no_list:
                index = available_cell_no_list.index(cell_no)
                all_soh_list.append(available_soh_array[index])
            else:
                all_soh_list.append(None)

        all_soh_array = np.array(all_soh_list)

        return all_soh_array, available_soh_array
    except Exception as e:
        log.logger.error(f'计算SOH时cal_soh函数出错，错误信息：{e}')

def get_soh_results(autocap_rlt_res):
    '''
    获取soh值
    '''
    try:
        cell_type = autocap_rlt_res['cell_type']
        standard_capacity = autocap_rlt_res['standard_capacity']
        cell_no_list = autocap_rlt_res['cell_no_list']
        cell_num = autocap_rlt_res['cell_num']
        autocap_data = autocap_rlt_res['autocap_data']

        all_cell_no_list = []  # 全部测试电芯号
        available_cell_no_list = []  # 全部有计算结果电芯号
        invalid_cell_no_list = []  # 无效电芯号
        all_capacity_list = []
        available_capacity_list = []
        error_records = ''  # 无计算结果的原因记录

        for idx, cell_no in enumerate(cell_no_list):
            # 提取数据
            cell_id = str(cell_no)
            cell_data = autocap_data[cell_id]
            dcap = cell_data['dcap']
            start_vol = cell_data['start_vol']
            end_vol = cell_data['end_vol']

            # 计算容量
            capacity, error_record = calculate_cell_capacities_by_voltages(dcap, start_vol, end_vol, cell_type, cell_id)
            error_records += error_record

            # 记录电芯号及容量
            all_cell_no_list.append(cell_no)
            all_capacity_list.append(capacity)

            # 有效电芯和无效电芯区分
            if capacity is not None:
                available_cell_no_list.append(cell_no)
                available_capacity_list.append(capacity)
            else:
                invalid_cell_no_list.append(cell_no)

        # 计算SOH
        all_soh_array, available_soh_array = cal_soh_scale_and_offset(all_cell_no_list, all_capacity_list, available_cell_no_list, available_capacity_list, standard_capacity, is_scale=True)

        return all_cell_no_list, all_soh_array, available_cell_no_list, available_soh_array, invalid_cell_no_list, error_records
    except Exception as e:
        log.logger.error(f'获取SOH结果get_soh_results函数出错，错误信息：{e}')

def add_out_dir_info(rlt_res, key, value, confidence, explanation):
    """
    向 rlt_res 中添加 key value and confidence explanation

    参数：
    rlt_res: 结果字典
    key: '充电起始总电压'或'充电结束总电压'
    value: 总电压值
    confidence: 可信度信息 [-10, 10]  正值代表可信度，值越大，可信度越高， 负值表示异常， 值越小，异常等级越高
    explanation : 对可信度值的解释
    """
    # 初始化指定的键
    rlt_res['out'][key] = []
    rlt_res['out'][key].append(value)
    rlt_res['out'][key].append(confidence)
    rlt_res['out'][key].append(explanation)

def get_output_json(all_cell_no_list, all_soh_array, available_cell_no_list, available_soh_array, invalid_cell_no_list, error_records):
    '''
    json形式
    '''
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

        advice = ''
        result_desc = ''

        if len(available_cell_no_list) > 0:
            max_soh = round(np.max(available_soh_array), 2)
            min_soh = round(np.min(available_soh_array), 2)
            # 数字，整数形式的电芯号，找到available_soh_array最大值对应位置available_cell_no_list值
            cell_no_max_soh = available_cell_no_list[np.argmax(available_soh_array)]
            cell_no_min_soh = available_cell_no_list[np.argmin(available_soh_array)]

            # soh异常电芯号，soh低于80且与中值差值大于5的电芯
            # 同时记录除异常电芯外最小SOH值
            soh_abnormal_cell_no = []
            normal_soh_values = []  # 存储正常电芯的SOH值
            median_soh = np.median(available_soh_array)

            for ii in range(len(available_soh_array)):
                current_soh = available_soh_array[ii]
                if current_soh < 80 and abs(current_soh - median_soh) > 5:
                    soh_abnormal_cell_no.append(ii + 1)  # 记录异常电芯号（+1表示从1开始编号）
                else:
                    normal_soh_values.append(current_soh)  # 收集正常电芯的SOH值

            # 计算除异常电芯外的最小SOH值
            if normal_soh_values:  # 确保存在正常电芯
                min_normal_soh = min(normal_soh_values)
            else:
                min_normal_soh = None  # 或根据需求设置默认值，如0

            # soh异常电芯号列表转为字符串
            soh_abnormal_cell_no_str = ','.join(str(i) for i in soh_abnormal_cell_no)
            # 测试SOH电芯号列表转为字符串
            available_cell_no_list_str = ','.join(str(i) for i in available_cell_no_list)
            
            # 测试电芯SOH列表转为字符串
            available_soh_list_str = ','.join(str(round(i,1)) for i in available_soh_array)
            
            # 建议及结果说明字段
            if len(available_cell_no_list) != len(all_cell_no_list):
                result_desc = f'本次进行SOH测试的电芯号依次为{all_cell_no_list_str}，其中部分电芯无有效计算结果，电芯号为{invalid_cell_no_list_str}。具有有效计算结果的电芯号为{available_cell_no_list_str}，对应SOH值（单位%）分别为{available_soh_list_str}。'
                if soh_abnormal_cell_no:
                    advice = f'本次进行SOH测试的全部电芯中，部分电芯无有效计算结果，有效计算结果中SOH异常电芯号为{soh_abnormal_cell_no_str}，其中第{cell_no_min_soh}节电芯SOH最小为{min_soh:.2f}%。建议更换异常电芯，维护后整组SOH最高可提升至{min_normal_soh:.2f}%。' 
                else:
                    advice = f'本次进行SOH测试的全部电芯中，部分电芯无有效计算结果，具有有效计算结果的电芯全部SOH正常，其中第{cell_no_min_soh}节电芯SOH最小为{min_soh:.2f}%，。'
            else:
                result_desc = f'本次进行SOH测试的电芯号依次为{all_cell_no_list_str}，全部具有有效计算结果，对应SOH值（单位%）分别为{available_soh_list_str}。'
                if soh_abnormal_cell_no:
                    advice = f'本次进行SOH测试的全部电芯中，SOH异常电芯号为{soh_abnormal_cell_no_str}，其中第{cell_no_min_soh}节电芯SOH最小为{min_soh:.2f}%。建议更换异常电芯，维护后整组SOH最高可提升至{min_normal_soh:.2f}%。'
                else:
                    advice = f'本次进行SOH测试的全部电芯中，全部SOH正常，其中第{cell_no_min_soh}节电芯SOH最小为{min_soh:.2f}%，第{cell_no_max_soh}节电芯SOH最大为{max_soh:.2f}%。'
        elif len(all_cell_no_list) > 0:
            result_desc = f'本次进行SOH测试的电芯号依次为{all_cell_no_list_str}，全部无有效计算结果。'
            advice = '本次进行SOH测试的全部电芯中，无有效计算结果，暂无维修建议。'
        else:
            result_desc = '本次没有进行SOH测试的电芯，或输入的电芯序号与实际测试通道号匹配有误，无有效计算结果。'
            advice = ''
            
        # 输出结果
        rlt_res = {
            "code_id": 2,
            "describe": "soh",
            "out": {},
            "summary": [],
            "table": [],
            "ErrorCode": [0, 0, '']
        }

        add_out_dir_info(rlt_res, 'SOH最高值', max_soh, '', '')
        add_out_dir_info(rlt_res, 'SOH最低值', min_soh, '', '')
        add_out_dir_info(rlt_res, 'SOH最高值电芯号', cell_no_max_soh, '', '')
        add_out_dir_info(rlt_res, 'SOH最低值电芯号', cell_no_min_soh, '', '')
        add_out_dir_info(rlt_res, 'SOH异常电芯列表', soh_abnormal_cell_no_str, '', '')
        add_out_dir_info(rlt_res, '测试SOH电芯号列表', all_cell_no_list_str, '', '')
        add_out_dir_info(rlt_res, '测试电芯SOH列表', all_soh_array_str, '', '')
        add_out_dir_info(rlt_res, 'SOH异常维修建议', advice, '', '')
        add_out_dir_info(rlt_res, 'SOH结果说明', result_desc, '', '')
        add_out_dir_info(rlt_res, 'SOH计算异常记录', error_records, '', '')
        # soh_info = {
        #     'SOH最高值': max_soh,
        #     'SOH最低值': min_soh,
        #     'SOH最高值电芯号': cell_no_max_soh,
        #     'SOH最低值电芯号': cell_no_min_soh,
        #     'SOH异常电芯列表': soh_abnormal_cell_no_str,
        #     '测试SOH电芯号列表': all_cell_no_list_str,
        #     '测试电芯SOH列表': all_soh_array_str,
        #     'SOH异常维修建议': advice,
        #     'SOH结果说明': result_desc,
        #     'SOH计算异常记录': error_records
        # }
        return rlt_res
    except Exception as e:
        log.logger.error(f'获取SOH结果get_output_json函数出错，错误信息：{e}')

def get_results(autocap_rlt_res):
    '''
    对外调用接口，返回json形式的soh结果
    '''
    try:
        all_cell_no_list, all_soh_array, available_cell_no_list, available_soh_array, invalid_cell_no_list, error_records = get_soh_results(autocap_rlt_res)
        soh_info = get_output_json(all_cell_no_list, all_soh_array, available_cell_no_list, available_soh_array, invalid_cell_no_list, error_records)
        return soh_info
    except Exception as e:
        log.logger.error(f'获取SOH结果get_results函数出错，错误信息：{e}')

# 使用示例
if __name__ == '__main__':
    data = {
        'cell_id': [1, 1, 1, 1, 1, 1, 1],
        'dcap': [6, 12, 115.83, 11.06, 9.96, 19.54, 10.1667],
        'start': [4136, 4085, 3990, 3436, 3070, 3070, 3485],
        'end': [4085, 3990, 3436, 3070, 3442, 3485, 3582],
        'cell_type': ['nmc', 'nmc', 'nmc', 'nmc', 'nmc', 'nmc', 'nmc']
    }
    df = pd.DataFrame(data)

    for ii in range(0, len(df['dcap'])):

        try:
            dcap = df['dcap'].iloc[ii]
            start_vol = df['start'].iloc[ii]
            end_vol = df['end'].iloc[ii]
            cell_type = df['cell_type'].iloc[ii]

            capacity = calculate_cell_capacities_by_voltages(dcap, start_vol, end_vol, cell_type, charge_cutoff_voltage=4.2, discharge_cutoff_voltage=2.5)

        except Exception as e:
            print(f'计算过程出错: {str(e)}')
