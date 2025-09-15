import os
import pandas as pd
import numpy as np
from tool_utils.pvlog import Logger

'''
获取充电的起始和结束电压，寻找对应ocv，然后计算soh
'''
level = 'error'
log = Logger('./logs/consist_balance_data.log', level=level)


class VoltageCapacityConsistency:
    def __init__(self, data_clean_rlt, balance_rlt_res):
        self.cell_type = data_clean_rlt['out']['battery_type'][0]
        self.standard_capacity = data_clean_rlt['out']['battery_capacity'][0]
        self.cell_no_list = balance_rlt_res['cell_no_list']
        self.cell_num = balance_rlt_res['cell_num']
        self.balance_data = balance_rlt_res['balance_data']
        self.ratio_diff_dcap_balance_percent_threshold = 10  # 容量偏差阈值，超过10%的电芯认为是异常
        self.diff_vol_mv_threshold = 0.1  # 电压偏差阈值，超过100mV的电压认为是异常

        # 全部结果初始化为清洗后结果
        self.all_balance_rlt = balance_rlt_res['balance_data'].copy()
        for i, cell_no in enumerate(self.cell_no_list):
            cell_id = str(cell_no)
            self.all_balance_rlt[cell_id]['start_vol_sort_index'] = None
            self.all_balance_rlt[cell_id]['end_vol_sort_index'] = None
            self.all_balance_rlt[cell_id]['diff_dcap'] = None
            self.all_balance_rlt[cell_id]['ratio_diff_dcap_balance_percent'] = None

        # 初始化一些结果变量为None
        self.max_diff_vol_start_mv = None
        self.max_diff_vol_end_mv = None
        self.min_volt_start = None
        self.max_volt_start = None
        self.min_cell_no_start_list = None
        self.max_cell_no_start_list = None
        self.min_volt_end = None
        self.max_volt_end = None
        self.min_cell_no_end_list = None
        self.max_cell_no_end_list = None
        
        self.start_vol_list = None  # 初始电压列表
        self.end_vol_list = None  # 结束电压列表

        self.dcap_array_ah = None
        self.max_diff_dcap_balance_ah = None
        self.max_ratio_diff_dcap_balance_percent = None
        self.ratio_diff_dcap_balance_percent = None
        self.abnormal_cell_no_list = None

        # 异常记录
        self.error_records = ''

    def get_min_max_voltages(self):
        '''
        获取起始时刻电压最高值，起始时刻电压最低值，对应电芯id，电压单位V
        获取电压排序列表
        计算初始最大压差，结束最大压差，单位mV
        '''
        try:
            min_volt_start = 5000
            max_volt_start = 0
            min_cell_no_start_list = []
            max_cell_no_start_list = []
            min_volt_end = 5000
            max_volt_end = 0
            min_cell_no_end_list = []
            max_cell_no_end_list = []
            abnormal_cell_no_list = []

            # 获取电压极值以及初始/结束电压列表
            start_vol_list = []
            end_vol_list = []
            for cell_no in self.cell_no_list:
                cell_id = str(cell_no)
                cell_data = self.balance_data[cell_id]
                start_vol = cell_data['start_vol']
                end_vol = cell_data['end_vol']

                # 电压列表
                start_vol_list.append(start_vol)
                end_vol_list.append(end_vol)

            # 以self.cell_no_list为准的升序排序后序号列表
            start_vol_list_sort_index = np.argsort(start_vol_list)
            end_vol_list_sort_index = np.argsort(end_vol_list)

            # 每个序号保存在self.all_balance_rlt[cell_id]
            for i, cell_no in enumerate(self.cell_no_list):
                cell_id = str(cell_no)
                self.all_balance_rlt[cell_id]['start_vol_sort_index'] = start_vol_list_sort_index[i]
                self.all_balance_rlt[cell_id]['end_vol_sort_index'] = end_vol_list_sort_index[i]
                # 获取电压极值，序号为最小值或者最大值的电芯号保存在对应列表
                if start_vol_list[i] == min(start_vol_list):
                    min_volt_start = min(min_volt_start, start_vol_list[i])
                    min_cell_no_start_list.append(cell_no)
                if start_vol_list[i] == max(start_vol_list):
                    max_volt_start = max(max_volt_start, start_vol_list[i])
                    max_cell_no_start_list.append(cell_no)
                if end_vol_list[i] == min(end_vol_list):
                    min_volt_end = min(min_volt_end, end_vol_list[i])
                    min_cell_no_end_list.append(cell_no)
                if end_vol_list[i] == max(end_vol_list):
                    max_volt_end = max(max_volt_end, end_vol_list[i])
                    max_cell_no_end_list.append(cell_no)

            # 计算压差
            max_diff_vol_start = (max(start_vol_list) - min(start_vol_list)) * 1000  # 单位mV
            max_diff_vol_end = (max(end_vol_list) - min(end_vol_list)) * 1000  # 单位mV

            # # 判断初始电压均值小于结束电压均值，认为是充电，结束时全部电压减去最高电压压差大于10mV的电芯保存至abnormal_cell_no_list
            # # 判断初始电压均值大于结束电压均值，认为是放电，结束时全部电压减去最低电压压差大于10mV的电芯保存至abnormal_cell_no_list
            # if np.mean(start_vol_list) < np.mean(end_vol_list):
            #     for i, cell_no in enumerate(self.cell_no_list):
            #         cell_id = str(cell_no)
            #         if end_vol_list[i] - max(end_vol_list) > self.diff_vol_mv_threshold:
            #             abnormal_cell_no_list.append(cell_no)
            # else:
            #     for i, cell_no in enumerate(self.cell_no_list):
            #         cell_id = str(cell_no)
            #         if end_vol_list[i] - min(end_vol_list) > self.diff_vol_mv_threshold:
            #             abnormal_cell_no_list.append(cell_no)

            # 判断电压偏低的电芯保存至abnormal_cell_no_list
            for i, cell_no in enumerate(self.cell_no_list):
                cell_id = str(cell_no)
                if max(end_vol_list) - end_vol_list[i]  > self.diff_vol_mv_threshold:
                    abnormal_cell_no_list.append(cell_no)
            
            # 保存在类
            self.min_volt_start = min_volt_start
            self.max_volt_start = max_volt_start
            self.min_cell_no_start_list = min_cell_no_start_list
            self.max_cell_no_start_list = max_cell_no_start_list
            self.min_volt_end = min_volt_end
            self.max_volt_end = max_volt_end
            self.min_cell_no_end_list = min_cell_no_end_list
            self.max_cell_no_end_list = max_cell_no_end_list
            self.start_vol_list = start_vol_list
            self.end_vol_list = end_vol_list
            self.max_diff_vol_start_mv = round(max_diff_vol_start, 2)
            self.max_diff_vol_end_mv = round(max_diff_vol_end, 2)
            self.abnormal_cell_no_list = abnormal_cell_no_list

            return
        except Exception as e:
            self.error_records += '电压极值获取失败。'
            raise RuntimeError(f'get_min_max_voltages函数出错,{e}') from e
    
    def get_dcap_balance(self):
        '''
        结束时刻容量偏差，全部减去最小值后除以标称容量，超过10%的电芯认为是异常
        '''
        try:
            dcap_list = []
            
            # 收集dcap数据
            for cell_no in self.cell_no_list:
                cell_id = str(cell_no)
                cell_data = self.balance_data[cell_id]
                dcap = cell_data['dcap']
                
                # 收集dcap数据
                dcap_list.append(dcap)

            # 计算容量偏差
            dcap_array = np.array(dcap_list)
            min_dcap = np.min(dcap_array)
            ratio_diff_dcap_balance = (dcap_array - min_dcap) / self.standard_capacity * 100  # 单位%

            # 寻找差值占比超过10%的电芯
            for i, cell_no in enumerate(self.cell_no_list):
                # 提取数据
                diff_dcap = dcap_array[i] - min_dcap
                ratio_diff_dcap_balance_percent = diff_dcap / self.standard_capacity * 100  # 单位%

                # 保存至全部结果dict
                cell_id = str(cell_no)
                self.all_balance_rlt[cell_id]['diff_dcap'] = diff_dcap
                self.all_balance_rlt[cell_id]['ratio_diff_dcap_balance_percent'] = ratio_diff_dcap_balance_percent
                    
            # 保存在类
            self.dcap_array_ah = dcap_array
            self.max_diff_dcap_balance_ah = max(dcap_array) - min_dcap
            self.max_ratio_diff_dcap_balance_percent = max(ratio_diff_dcap_balance)
            self.ratio_diff_dcap_balance_percent = ratio_diff_dcap_balance
            
            return
        except Exception as e:
            self.error_records += '容量偏差获取失败。'
            raise RuntimeError(f'get_dcap_balance函数出错,{e}') from e

    def get_output_json(self, rlt_res):
        '''
        json形式，包含以下内容：
        全部结果包含每个电芯结果，内部包含初始电压、结束电压、初始电压排序、结束电压排序、容差、容差占比
        初始最大压差、结束最大压差
        最高单体电压值、最高单体电芯号列表、最低单体电压值、最低单体电芯号列表、电压一致性异常电芯列表、均衡起始电压列表、均衡结束电压列表
        '''
        try:

            # 判断最高单体电压值、最高单体电芯号、最低单体电压值、最低单体电芯号的列表是否有效，有效全部转化为字符串，无效则赋值为''
            min_cell_no_start_str = ','.join(str(i) for i in self.min_cell_no_start_list) if self.min_cell_no_start_list is not None else ''
            max_cell_no_start_str = ','.join(str(i) for i in self.max_cell_no_start_list) if self.max_cell_no_start_list is not None else ''
            min_cell_no_end_str = ','.join(str(i) for i in self.min_cell_no_end_list) if self.min_cell_no_end_list is not None else ''
            max_cell_no_end_str = ','.join(str(i) for i in self.max_cell_no_end_list) if self.max_cell_no_end_list is not None else ''

            # 判断电压一致性异常电芯列表是否有效，有效全部转化为字符串，无效则赋值为''
            abnormal_cell_no_str = ','.join(str(i) for i in self.abnormal_cell_no_list) if self.abnormal_cell_no_list is not None else ''

            # 判断容量变化、容量偏差占比列表是否有效，有效全部转化为字符串，无效则赋值为'' 
            dcap_array_ah_str = ','.join(str(i) for i in self.dcap_array_ah) if self.dcap_array_ah is not None else ''
            ratio_diff_dcap_balance_percent_str = ','.join(str(i) for i in self.ratio_diff_dcap_balance_percent) if self.ratio_diff_dcap_balance_percent is not None else ''
            

            # 判断均衡起始电压列表、均衡结束电压列表是否有效，有效全部转化为字符串，无效则赋值为''
            start_vol_str = ','.join(str(i) for i in self.start_vol_list) if self.start_vol_list is not None else ''
            end_vol_str = ','.join(str(i) for i in self.end_vol_list) if self.end_vol_list is not None else ''

            # 全部测试电芯转化为字符串
            cell_no_str = ','.join(str(i) for i in self.cell_no_list) if len(self.cell_no_list) > 0 else ''

            # 建议及结果说明字段
            if len(self.cell_no_list) > 0:
                result_desc = f'本次进行均衡测试的电芯号依次为{cell_no_str}，其中均衡前第{max_cell_no_start_str}节电芯电压最高为{self.max_volt_start}V，\
                    第{min_cell_no_start_str}节电芯电压最低为{self.min_volt_start}V，\
                    均衡后第{max_cell_no_end_str}节电芯电压最高为{self.max_volt_end}V，\
                    第{min_cell_no_end_str}节电芯电压最低为{self.min_volt_end}V。\
                    均衡前最大压差为{self.max_diff_vol_start_mv}mV，均衡后最大压差为{self.max_diff_vol_end_mv}mV。{self.max_diff_vol_end_mv}mV，\
                    均衡前容量偏差最大为{self.max_diff_dcap_balance_ah}Ah，占标称容量比例为{self.max_ratio_diff_dcap_balance_percent}%。\
                    详细信息如下：均衡前电压依次为{start_vol_str}V，均衡后电压依次为{end_vol_str}V，\
                    均衡充入/放出容量依次为{dcap_array_ah_str}，均衡前容量偏差占标称容量比例依次为{ratio_diff_dcap_balance_percent_str}%。\
                    若有同个电芯进行了多次测试，以最后一次测试结果为准。'
                if len(self.abnormal_cell_no_list) > 0:
                    advice = f'均衡结束时仍有一些电压偏低哦，电芯号为{abnormal_cell_no_str}，可以试试在低SOC均衡。'
                else:
                    advice = f'均衡前最大压差为{self.max_diff_vol_start_mv}mV，均衡后最大压差为{self.max_diff_vol_end_mv}mV，\
                        单电芯均衡容量最多的为{max(self.dcap_array_ah)}Ah，最少的为{min(self.dcap_array_ah)}Ah。'
            else:
                result_desc = '本次没有进行均衡测试的电芯，或输入的电芯序号与实际测试通道号匹配有误，无有效计算结果。'
                advice = ''
                
            # 初始最大压差、结束最大压差保存至结果字典
            self.add_out_dir_info(rlt_res, '一致性评估全部结果', self.all_balance_rlt, '', '')
            self.add_out_dir_info(rlt_res, '初始最大压差', self.max_diff_vol_start_mv, '', '')
            self.add_out_dir_info(rlt_res, '结束最大压差', self.max_diff_vol_end_mv, '', '')
            self.add_out_dir_info(rlt_res, '容差最大值', self.max_diff_dcap_balance_ah, '', '')
            self.add_out_dir_info(rlt_res, '容差占比最大值', self.max_ratio_diff_dcap_balance_percent, '', '')
            self.add_out_dir_info(rlt_res, '均衡前最高单体电压值', self.max_volt_start, '', '')
            self.add_out_dir_info(rlt_res, '均衡前最高单体电芯号列表', max_cell_no_start_str, '', '')
            self.add_out_dir_info(rlt_res, '均衡前最低单体电压值', self.min_volt_start, '', '')
            self.add_out_dir_info(rlt_res, '均衡前最低单体电芯号列表', min_cell_no_start_str, '', '')
            self.add_out_dir_info(rlt_res, '均衡后最高单体电压值', self.max_volt_end, '', '')
            self.add_out_dir_info(rlt_res, '均衡后最高单体电芯号列表', max_cell_no_end_str, '', '')
            self.add_out_dir_info(rlt_res, '均衡后最低单体电压值', self.min_volt_end, '', '')
            self.add_out_dir_info(rlt_res, '均衡后最低单体电芯号列表', min_cell_no_end_str, '', '')
            self.add_out_dir_info(rlt_res, '电压一致性异常电芯列表', abnormal_cell_no_str, '', '')
            self.add_out_dir_info(rlt_res, '均衡起始电压列表', start_vol_str, '', '')
            self.add_out_dir_info(rlt_res, '均衡结束电压列表', end_vol_str, '', '')
            self.add_out_dir_info(rlt_res, '容差占比', ratio_diff_dcap_balance_percent_str, '', '')
            self.add_out_dir_info(rlt_res, '均衡容量', dcap_array_ah_str, '', '')
            self.add_out_dir_info(rlt_res, '一致性计算异常说明', self.error_records, '', '')
            
            # 说明及建议
            self.add_out_dir_info(rlt_res, '说明', result_desc, '', '')
            self.add_out_dir_info(rlt_res, '建议', advice, '', '')

            return rlt_res
        except Exception as e:
            self.error_records += '输出json结果获取失败。'
            raise RuntimeError(f'get_output_json函数出错,{e}') from e

    @staticmethod
    def add_out_dir_info(rlt_res, key, value, confidence, explanation):
        """向结果字典中添加字段"""
        rlt_res['out'][key] = [value, confidence, explanation]

    def invalid_result_handle(self, rlt_res):
        """处理无效结果"""
        self.add_out_dir_info(rlt_res, '一致性评估全部结果', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '初始最大压差', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '结束最大压差', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '容差最大值', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '容差占比最大值', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡前最高单体电压值', 'N/A', '', '')        
        self.add_out_dir_info(rlt_res, '均衡前最高单体电芯号列表', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡前最低单体电压值', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡前最低单体电芯号列表', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡后最高单体电压值', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡后最高单体电芯号列表', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡后最低单体电压值', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡后最低单体电芯号列表', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '电压一致性异常电芯列表', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡起始电压列表', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡结束电压列表', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '容差占比', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '均衡容量', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '一致性计算异常说明', self.error_records, '', '')
        self.add_out_dir_info(rlt_res, '说明', 'N/A', '', '')
        self.add_out_dir_info(rlt_res, '建议', 'N/A', '', '')

def run(balance_rlt_res, data_clean_rlt):
    '''
    对外调用接口，返回json形式的soh结果
    '''
    # 输出结果
    rlt_res = {
        "code_id": 2,
        "describe": "soh",
        "out": {},
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }

    try:
        consis = VoltageCapacityConsistency(data_clean_rlt, balance_rlt_res)
        consis.get_min_max_voltages()
        consis.get_dcap_balance()
        consistency_info = consis.get_output_json(rlt_res)
        return consistency_info
    except Exception as e:
        rlt_res['ErrorCode'][0] = 3001
        consis.invalid_result_handle(rlt_res)
        log.logger.error(f'获取一致性结果异常：{e}')
        return rlt_res
