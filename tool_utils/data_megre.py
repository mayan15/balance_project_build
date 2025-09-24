# -*- coding: utf-8 -*-
import os
import struct, binascii, traceback
import shutil

import re
import glob
import pandas as pd
from datetime import datetime

from tool_utils.pvlog import Logger
level = "debug"
# 记录日志文件
data_log = Logger('./logs/data_merge.log', level=level)
# 内阻数据整合成一个文件 
class DataMerge:
    def __init__(self, filepath):
        self.filepath = filepath

    def extract_cell_id_mapping(self, filename, test_type='pulse'):
        """
        从文件名中提取每个通道（pulse_vol_1~8）对应的cell_id列表
        返回长度为8的列表，表示通道1~8对应的cell_id
        """
        # 匹配 pulse_开头，后面跟 8 个 _xx 的 cell_id，再接时间戳
        # match = re.search(r'pulse_((?:\d+_){7}\d+)_\d{14}\.csv', filename)
        if test_type == 'pulse':
            match = re.search(r'pulse_((?:\d+_){7}\d+)_([\d]{14})\.csv', filename)
        elif test_type == 'autocap':
            match = re.search(r'autocap_((?:\d+_){7}\d+)_([\d]{14})\.csv', filename)
        elif test_type == 'balance':
            match = re.search(r'balance_((?:\d+_){7}\d+)_([\d]{14})\.csv', filename)
        else:
            raise ValueError(f"未知的测试类型: {test_type}")
        if not match:
            raise ValueError(f"无法从文件名提取电池号映射: {filename}")

        cell_id_str = match.group(1)  # 提取 8 个数字部分
        timestamp = match.group(2) # 提前时间戳
        cell_id_list = [int(s) for s in cell_id_str.split('_')]

        if len(cell_id_list) != 8:
            raise ValueError(f"文件名中通道映射数量不是8个: {filename}")
        return cell_id_list, timestamp
    
    def _extract_channel_data(self, df, cell_id_list):
        """
        从DataFrame中提取每个通道（pulse_vol_1~8）的数据
        返回记录列表和未收到数据的cell_id列表
        """
        records = []
        less_cell =[] 
        for i in range(8):
            vol_col = f'pulse_vol_{i + 1}'       # i的范围 0-7 
            cur_col = f'pulse_cur_{ i + 1}'
            collect_piont_col = f'pulse_collect_point_{i + 1}'
            
            cell_id =  cell_id_list[i]
            if cell_id == 0:
                continue  # 跳过未接入的通道
            pulse_repeat_cnt = f'pulse_total_cnt_{i + 1}'
            pulse_present_cnt = f'pulse_rev_cnt_{i + 1}'
            pulse_idle_time = f'pulse_idle_time_{i + 1}'
            pulse_on_time = f'pulse_on_time_{i + 1}'
            temp_records = []
            for val , cur , collect_piont, pulse_present_cnt , total_cnt , idle_time, on_time in zip(df[vol_col], df[cur_col], df[collect_piont_col], df[pulse_present_cnt], df[pulse_repeat_cnt], df[pulse_idle_time], df[pulse_on_time]):  
                if isinstance(val, str) and val.strip() != '[]':
                    temp_records.append({
                        'cell_id': cell_id,
                        'pulse_repeat_cnt': total_cnt,
                        'pulse_present_cnt': pulse_present_cnt,
                        'pulse_collect_point': collect_piont,
                        'pulse_idle_time': idle_time,
                        'pulse_on_time':on_time, 
                        'pulse_cur': cur,
                        'pulse_vol': val,
                    })
                    # pulse_present_cnt += 1
            # 对该通道的记录按 pulse_present_cnt 排序
            temp_records.sort(key=lambda x: x['pulse_present_cnt'])
            records.extend(temp_records)
            # 判断当前cell是否收到了完整的数据, 若未收到, 则将cell_id添加到less_cell列表中
            if pulse_present_cnt < int(total_cnt):
               less_cell.append(cell_id)

        return records, less_cell

    def out_all_pulse_data(self):
        # 搜索所有pulse_*.csv文件
        file_path = self.filepath + '/pulse_*.csv'
        csv_files = glob.glob(file_path)
        all_records = []
        all_less_cell_id = []
        all_timestamps = []
        for file in csv_files:
            try:
                cell_id_list, timestamp = self.extract_cell_id_mapping(os.path.basename(file))
                df = pd.read_csv(file)
                records, less_cell_id  = self._extract_channel_data(df, cell_id_list)
                all_records.extend(records)
                all_less_cell_id.extend(less_cell_id)
                all_timestamps.append(timestamp)
                
            except Exception as e:
                data_log.logger.error(f"merge pulse data error: {traceback.print_exc()}")

        # 合并所有结果并导出
        final_df = None
        if all_records:
            final_df = pd.DataFrame(all_records)
            # final_df.to_csv(self.filepath + '/' + all_timestamps[0] +'_'+ all_timestamps[-1] + '_pulseAllDate.csv', index=False)
        return  final_df 

    # 提取容量计算数据
    def out_all_capacity_data(self):
        autocap_data_raw = {'df_list':[], 'cell_no_lists': [], 'timestamp_list': []}

        file_path = self.filepath + '/autocap_*.csv'
        csv_files = glob.glob(file_path)

        for file in csv_files:
            try:
                cell_id_list, timestamp = self.extract_cell_id_mapping(os.path.basename(file), test_type='autocap')
                df = pd.read_csv(file)
                autocap_data_raw['cell_no_lists'].append(cell_id_list)
                autocap_data_raw['timestamp_list'].append(timestamp)

                # 收集原始数据
                autocap_data_raw['df_list'].append(df)
            except Exception as e:
                data_log.logger.error(f"文件名 {file} out_all_capacity_data提取数据时异常{e}") 
        return autocap_data_raw
    
    def out_all_balance_data(self):
        autocap_data_raw = {'df_list':[], 'cell_no_lists': [], 'timestamp_list': []}

        file_path = self.filepath + '/balance_*.csv'
        csv_files = glob.glob(file_path)

        for file in csv_files:
            try:
                cell_id_list, timestamp = self.extract_cell_id_mapping(os.path.basename(file), test_type='balance')
                df = pd.read_csv(file)
                autocap_data_raw['cell_no_lists'].append(cell_id_list)
                autocap_data_raw['timestamp_list'].append(timestamp)

                # 收集原始数据
                autocap_data_raw['df_list'].append(df)
            except Exception as e:
                data_log.logger.error(f"文件名 {file} out_all_balance_data提取数据时异常{e}") 
        return autocap_data_raw
        

# # 提取容量计算数据
# def select_capacity_data(autocap_data_raw, filename, df):
#     # 按前缀分类存储数据
#     if filename.startswith('autocap'):

#         # 收集电芯号
#         match = re.search(r"autocap_((?:\d+_)+)\d{14}\.csv", filename)
#         if match:
#             nums_str = match.group(1).strip("_")
#             cell_ids = list(map(int, nums_str.split("_")))
#             autocap_data_raw['cell_no_list'].append(cell_ids)

#             # 收集原始数据
#             autocap_data_raw['df_list'].append(df)

#         else:
#             data_log.logger.error(f"文件名 {filename} 格式错误，无法提取电芯号") 