import math

import numpy as np
import pandas as pd
import time
import pickle
import traceback
from datetime import datetime

from tool_utils.pvlog import Logger
level = "error"
# 记录日志文件
data_clean_log = Logger('./logs/data_clean_overview.log', level=level)

def check_invalid_col(df, column_names):
    """
    检查 DataFrame 中是否存在指定的列名。

    :param df: 输入的 DataFrame
    :param column_names: 要检查的列名列表
    :return: 存在的列名组成的列表
    """
    # 找到存在于 DataFrame 中的列名
    existing_columns_lv1 = [col for col in column_names if col in df.columns]
    if len(existing_columns_lv1) != len(column_names):
        return False
    return True


def judge(df, column_name, limit_max, limit_min, need=True):
    """
        检查 DataFrame 中是否存在指定的列名, 并判断该列存在的值是否正常

        :param df: 输入的 DataFrame
        :param column_names: 要检查的列名列表
        :limit_max: 最大有效值
        :limit_min: 最小有效值
        :need: 某些值不需要判断是否属于正常范围
    """
    confidence = ''
    explanation = ''
    value = '/'

    if column_name in df.columns and pd.isna(df[column_name][0]) == False:
        value = df[column_name][0]
        if need:
            if value > limit_max:
                confidence = '↑'
            elif value < limit_min:
                confidence = '↓'
    else:
        confidence = 'N/A'
        explanation = '缺乏关键数值，不支持分析'
    return value, confidence, explanation


def judge_value(value, limit_max, limit_min):
    """
        检查数值是否符合需求

        :param value: 需要判断的数值
        :param limit_max: 最大有效值
        :param limit_min: 最小有效值
    """
    confidence = ''
    explanation = ''
    
    if value > limit_max:
        confidence = '↑'
    elif value < limit_min:
        confidence = '↓'
    return confidence, explanation


def data_clean(df):
    """
        对原始数据进行清洗和画像， 清洗完成后， 返回的数据均为带时间序列的数据

        :param df: 输入的 DataFrame

        :return: [ErrorCode, df_clean]

    """
    
    try:
        # df.drop_duplicates(['date_time'], keep='first', inplace=True)
        # df 中选择特定的列，并删除包含缺失值（NaN）的行
        cleaned_df = df
        return [0, cleaned_df]
    except Exception as e:
        print(f"data_clean failed: { traceback.print_exc()}")
        return [-3, df]


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
    # 添加总电压、默认值和可信度
    rlt_res['out'][key].append(value)
    rlt_res['out'][key].append(confidence)
    rlt_res['out'][key].append(explanation)


from ast import literal_eval
from collections import defaultdict

def merge_pulse_data(df_cleaned):
    # df_cleaned["pulse_cur"] = df_cleaned["pulse_cur"].apply(literal_eval)
    df_cleaned["pulse_cur"] = df_cleaned["pulse_cur"].astype(int)
    df_cleaned["pulse_vol"] = df_cleaned["pulse_vol"].apply(literal_eval)

    # 初始化结果字典
    result = defaultdict(lambda: {"pulse_cur": [], "pulse_vol": [], "time":[]})
    df_cleaned["pulse_present_cnt"] = df_cleaned["pulse_present_cnt"].astype(int) 
    df_cleaned['pulse_repeat_cnt'] = df_cleaned["pulse_repeat_cnt"].astype(int)
    df_cleaned['pulse_idle_time'] = df_cleaned["pulse_idle_time"].astype(int)
    df_cleaned['pulse_on_time'] = df_cleaned["pulse_on_time"].astype(int)
    # 按 cell_id 分组
    for cell_id, group in df_cleaned.groupby("cell_id"):
        # 按原始顺序遍历，不排序
        group_sorted = group.sort_index()

        cur_cycle, vol_cycle = [], []
        expected = 1
        
        for _, row in group_sorted.iterrows():
            cnt = row["pulse_present_cnt"]
            if cnt == expected:
                # 20250827 --- 将单个cur 扩展成1300的列表
                cur = []
                cur.append(0)
                cur[1:] = [row["pulse_cur"]] * (row["pulse_on_time"]*10-1) # cur is not zero
                cur[row["pulse_on_time"]*10:] = [0]*(row["pulse_idle_time"]*10)  # cur is zero

                cur_cycle.extend(cur)
                vol_cycle.extend(row["pulse_vol"])

                if expected == row["pulse_repeat_cnt"]:  # 完成一轮
                    result[cell_id]["pulse_cur"].append(cur_cycle)
                    result[cell_id]["pulse_vol"].append(vol_cycle)
                    # 重置
                    cur_cycle, vol_cycle = [], []
                    expected = 1
                else:
                    expected += 1
            else:
                # 数据断裂，重置
                cur_cycle, vol_cycle = [], []
                expected = 1  
        # 按照0.1s间隔，生成时间序列
        end = 0.1 * (len(result[cell_id]["pulse_vol"][0]) - 1)  # 计算终点值：519.9
        arr = np.linspace(0, end, num=len(result[cell_id]["pulse_vol"][0]))  # 关键函数[1,5](@ref)
        result[cell_id]["time"] = arr
    return result     


class AutocapBalanceDataProcessor:
    def __init__(self):
        self.vol_data_prefix = 'vol_'
        self.ah_data_prefix = 'bms_ah_'
        self.cur_data_prefix = 'cur_'
        self.error_code_col = 'error_code_'

    def data_clean(self, df):
        """
        对原始数据进行清洗，去掉error_code不为0的行，存在空值的行，电压不符合合理范围的行，返回清洗后的完整数据

        :param df: 输入的 DataFrame
        :return: [ErrorCode, df_clean]
        """
        # 任意一步结果为空则返回错误，否则继续
        try:
            
            if df.empty:
                return ['初始数据为空。', df]
            
            # 1.确认计算需要的列是否存在，全部都要在df中
            vol_columns = [col for col in df.columns if col.startswith(self.vol_data_prefix)]
            ah_columns = [col for col in df.columns if col.startswith(self.ah_data_prefix)]
            cur_columns = [col for col in df.columns if col.startswith(self.cur_data_prefix)]
            error_code_columns = [col for col in df.columns if col.startswith(self.error_code_col)]
            if not (vol_columns and ah_columns and cur_columns and error_code_columns):
                return ['缺少必要的列。', df]

            # 2.电压如果大于10需要除以1000，容量除以100
            for vol_col in vol_columns:
                df[vol_col] = df[vol_col] / 1000
            for ah_col in ah_columns:
                df[ah_col] = df[ah_col] / 100
            
            # 3. 去掉error_code不为0的行
            error_code_columns = [col for col in df.columns if col.startswith(self.error_code_col)]
            df = df[df[error_code_columns].eq(0).all(axis=1)]
            if df.empty:
                return ['数据中所有行的error_code均不全部为0。', df]

            # 4. 电压、电流、容量任一列存在空值的行
            df = df.dropna(subset=vol_columns + ah_columns + cur_columns)
            if df.empty:
                return ['数据去掉存在空值行后为空。', df]

            # 5. 电压不符合合理范围的行
            for vol_col in vol_columns:
                df = df[df[vol_col].between(-0.1, 5)]
            if df.empty:
                return ['数据中去掉电压不在合理范围内的行后为空。', df]
            
            return ['正常', df]
        except Exception as e:
            data_clean_log.logger.error(f"data_clean failed: {e}")
            return ['清洗数据时报错', df]


    def autocap_data_process(self, autocap_data_raw):
        """
        提取计算容量需要的信息，并返回字典

        :param autocap_data_raw: {
            'df_list': [DataFrame, ...],         # 原始数据列表
            'cell_no_lists': [[cell_no, ...], ...]  # 每个文件对应的电芯号列表
        }
        :return: {
            'cell_no_list': [有效电芯号列表],
            'cell_num': 有效电芯数量,
            'autocap_data': {cell_id: {start_vol, end_vol, dcap}, ...},
            'error_logs': {file_x: 错误原因, ...}
        }
        """
        autocap_data = {}         # 存放每个电芯的起始电压、结束电压、容量
        autocap_df_list = autocap_data_raw['df_list']
        cell_no_lists = autocap_data_raw['cell_no_lists']
        timestamp_list = autocap_data_raw['timestamp_list']
        available_cell_no_set = set()   # 记录所有有效电芯号（去重）
        error_logs = {}                 # 记录每个文件的清洗错误信息

        # 遍历每个输入文件的数据
        for idx_file, df in enumerate(autocap_df_list):
            # 筛选电压列和容量列
            vol_columns = [col for col in df.columns if col.startswith(self.vol_data_prefix)]
            ah_columns = [col for col in df.columns if col.startswith(self.ah_data_prefix)]
            current_columns = [col for col in df.columns if col.startswith('cur_')]

            # 获取测试时间
            timestamp = timestamp_list[idx_file]

            # 先清洗数据
            rlt, df_clean = self.data_clean(df)
            if rlt != '正常':
                # 保存错误日志，标明是哪个文件的问题
                error_logs[f'file_{idx_file}'] = rlt
                continue

            # 遍历该文件对应的电芯号
            for idx_channel, cell_no in enumerate(cell_no_lists[idx_file]):
                # 电芯号为 0 或者索引超过列数则跳过
                if (
                    cell_no == 0
                    or idx_channel >= len(vol_columns)
                    or idx_channel >= len(ah_columns)
                ):
                    continue

                # 如果该电芯号计算过了，则比较当前时间戳和历史i计算时间戳，以测试时间较晚的为准
                if cell_no in autocap_data:
                    if timestamp < autocap_data[cell_no]['timestamp']:
                        continue

                # 直接取清洗后的电压、容量序列
                vol_series = df_clean[vol_columns[idx_channel]]
                ah_series = df_clean[ah_columns[idx_channel]]
                current_series = df_clean[current_columns[idx_channel]]

                # 记录电芯号（转字符串作为 key）
                cell_id = str(cell_no)

                # 找到电压最小值的索引
                min_vol_index = vol_series.idxmin()
                # 找到电压最小值之后且电流大于0的第一个电压值的索引
                valid_indices = vol_series.index[min_vol_index+1:]
                start_vol_index = None
                for idx in valid_indices:
                    if current_series.loc[idx] > 0:
                        start_vol_index = idx
                        break
                if start_vol_index is None:
                    # 如果没有找到符合条件的电压值，跳过该电芯号
                    continue

                # 计算充电容量为start_vol_index+1后ah_series的最大值
                dcap = ah_series.loc[start_vol_index+1:].max()

                # 保存该电芯的起始电压、结束电压、容量
                autocap_data[cell_id] = {
                    'timestamp': timestamp,
                    'start_vol': vol_series.loc[start_vol_index],  # 起始电压为电压最小值之后且电流大于0的第一个电压值
                    'end_vol': vol_series.iloc[-1],   # 结束电压
                    'dcap': dcap        # 最终容量（已在 data_clean 处理过）
                }

                # 将该电芯号加入有效集合
                available_cell_no_set.add(cell_no)

        # --- 汇总最终结果 ---
        autocap_rlt_res = {
            'cell_no_list': sorted(available_cell_no_set),  # 有效电芯号排序
            'cell_num': len(available_cell_no_set),         # 有效电芯数
            'autocap_data': autocap_data,                   # 每个电芯的详细数据
            'error_logs': error_logs                        # 错误日志
        }

        return autocap_rlt_res


    def balance_data_process(self, balance_data_raw):
        """
        提取计算电压一致性需要的信息，并返回字典

        :param balance_data_raw: {
            'df_list': [DataFrame, ...],         # 原始数据列表
            'cell_no_lists': [[cell_no, ...], ...]  # 每个文件对应的电芯号列表
        }
        :return: {
            'cell_no_list': [有效电芯号列表],
            'cell_num': 有效电芯数量,
            'balance_data': {cell_id: {start_vol, end_vol, dcap}, ...},
            'error_logs': {file_x: 错误原因, ...}
        }
        """
        balance_data = {}         # 存放每个电芯的起始电压、结束电压、容量
        balance_df_list = balance_data_raw['df_list']
        cell_no_lists = balance_data_raw['cell_no_lists']
        timestamp_list = balance_data_raw['timestamp_list']
        available_cell_no_set = set()   # 记录所有有效电芯号（去重）
        error_logs = {}                 # 记录每个文件的清洗错误信息

        # 遍历每个输入文件的数据
        for idx_file, df in enumerate(balance_df_list):
            # 筛选电压列和容量列
            vol_columns = [col for col in df.columns if col.startswith(self.vol_data_prefix)]
            ah_columns = [col for col in df.columns if col.startswith(self.ah_data_prefix)]

            # 获取测试时间
            timestamp = timestamp_list[idx_file]

            # 先清洗数据
            rlt, df_clean = self.data_clean(df)
            if rlt != '正常':
                # 保存错误日志，标明是哪个文件的问题
                error_logs[f'file_{idx_file}'] = rlt
                continue

            # 遍历该文件对应的电芯号
            for idx_channel, cell_no in enumerate(cell_no_lists[idx_file]):
                # 电芯号为 0 或者索引超过列数则跳过
                if (
                    cell_no == 0
                    or idx_channel >= len(vol_columns)
                    or idx_channel >= len(ah_columns)
                ):
                    continue

                # 如果该电芯号计算过了，则比较当前时间戳和历史i计算时间戳，以测试时间较晚的为准
                if cell_no in balance_data:
                    if timestamp < balance_data[cell_no]['timestamp']:
                        continue

                # 直接取清洗后的电压、容量序列
                vol_series = df_clean[vol_columns[idx_channel]]
                ah_series = df_clean[ah_columns[idx_channel]]

                # 记录电芯号（转字符串作为 key）
                cell_id = str(cell_no)

                # 保存该电芯的起始电压、结束电压、容量
                balance_data[cell_id] = {
                    'timestamp': timestamp,
                    'start_vol': vol_series.iloc[0],   # 起始电压（已在 data_clean 转换过单位）
                    'end_vol': vol_series.iloc[-1],   # 结束电压
                    'dcap': ah_series.iloc[-1]        # 最终容量（已在 data_clean 处理过）
                }

                # 将该电芯号加入有效集合
                available_cell_no_set.add(cell_no)

        # --- 汇总最终结果 ---
        balance_rlt_res = {
            'cell_no_list': sorted(available_cell_no_set),  # 有效电芯号排序
            'cell_num': len(available_cell_no_set),         # 有效电芯数
            'balance_data': balance_data,                   # 每个电芯的详细数据
            'error_logs': error_logs                        # 错误日志
        }

        return balance_rlt_res


def run(df_pulse, dict_autocap, dict_balance, cell_config):
    """
        对原始数据进行清洗以及数据画像统计

        :param df_origin: pd读取的某一个csv文件数据
        :param picture_save_path: 数据画像生成的图片保存路径
        :param pickle_save_path:  数据处理结果，用于其他程序读取
        :return:    正常:  rlt_res = {
                            "code_id": 1,
                            "describe": "data clean and overview",
                            "out": {},
                            "summary": [],
                            "table": [],
                            "ErrorCode": [0, 0, '']}，
                               
                    错误：  rlt_res，ErrorCode[0]: error id; ErrorCode[2]: 详细的报错说明
                           [] 空列表
    """
    rlt_res = {
        "code_id": 1,
        "describe": "data clean and overview",
        "out": {},
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }
    try:
        # -------- 获取电芯相关信息  ---------#
        add_out_dir_info(rlt_res, 'vin', cell_config['data_config']['vin'], '', '')
        add_out_dir_info(rlt_res, 'mileage', cell_config['data_config']['mileage'], '', '')
        add_out_dir_info(rlt_res, 'device_id', cell_config['data_config']['device_id'], '', '')
        
        add_out_dir_info(rlt_res, 'battery_capacity', cell_config['data_config']['battery_capacity'], '', '') 

        if cell_config['data_config']['battery_type'] == "磷酸铁锂":
            add_out_dir_info(rlt_res, 'battery_type', 'LFP', '', '')
        elif cell_config['data_config']['battery_type'] == "三元锂":
            add_out_dir_info(rlt_res, 'battery_type', 'NCM', '', '')

        # add_out_dir_info(rlt_res, 'battery_voltage', cell_config['data_config']['battery_voltage'], '', '')
        add_out_dir_info(rlt_res, 'battery_rata_energy', cell_config['data_config']['battery_rata_energy'], '', '')
        add_out_dir_info(rlt_res, 'battery_manufacture_date', cell_config['data_config']['battery_manufacture_date'], '', '')
        add_out_dir_info(rlt_res, 'battery_manufacture', cell_config['data_config']['battery_manufacture'], '', '')

        # -------- pulse data  ---------#
        # 判断pulse data 是否符合需求， 并对数据做进一步处理
        if df_pulse is not None:
            check_column = ['cell_id', 'pulse_repeat_cnt', 'pulse_present_cnt', 'pulse_cur', 'pulse_vol']
            rlt = check_invalid_col(df_pulse, check_column)
            if not rlt:
                add_out_dir_info(rlt_res, 'pulse_data', None, '', '')
            all_pulse_data = merge_pulse_data(df_pulse)
            add_out_dir_info(rlt_res, 'pulse_data', all_pulse_data, '', '')

        else:
            add_out_dir_info(rlt_res, 'pulse_data', None, '', '')

        # 实例化AutocapBalanceDataProcessor
        abdp_obj = AutocapBalanceDataProcessor()

        # --------容量检测数据 ---------#
        # 提取电压、容量数据，添加到rlt_res
        autocap_rlt_res = abdp_obj.autocap_data_process(dict_autocap)
        add_out_dir_info(rlt_res, 'autocap_data', autocap_rlt_res, '', '')

        # --------电压一致性数据 ---------#
        # 提取电压、容量数据，添加到rlt_res
        balance_rlt_res = abdp_obj.balance_data_process(dict_balance)
        add_out_dir_info(rlt_res, 'balance_data', balance_rlt_res, '', '')
        
        return rlt_res
    except Exception as e:
        rlt_res['ErrorCode'][0] = -99
        rlt_res['ErrorCode'][2] = f'数据清洗报错{traceback.print_exc()}'
        return rlt_res, []
