# -*- coding: utf-8 -*-

import math

import numpy as np
import pandas as pd
import time
import pickle
import traceback
from datetime import datetime

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

def merge_plus_data(df_cleaned):
    # df_cleaned["plus_cur"] = df_cleaned["plus_cur"].apply(literal_eval)
    df_cleaned["plus_cur"] = df_cleaned["plus_cur"].astype(int)
    df_cleaned["plus_vol"] = df_cleaned["plus_vol"].apply(literal_eval)

    # 初始化结果字典
    result = defaultdict(lambda: {"plus_cur": [], "plus_vol": [], "time":[]})
    df_cleaned["plus_present_cnt"] = df_cleaned["plus_present_cnt"].astype(int) 
    # 按 cell_id 分组
    for cell_id, group in df_cleaned.groupby("cell_id"):
        # 按原始顺序遍历，不排序
        group_sorted = group.sort_index()

        cur_cycle, vol_cycle = [], []
        expected = 1
        
        for _, row in group_sorted.iterrows():
            cnt = row["plus_present_cnt"]
            if cnt == expected:
                # 20250827 --- 将单个cur 扩展成1300的列表
                cur = []
                cur.append(0)
                cur[1:] = [row["plus_cur"]] * (60*10-1) # cur is not zero
                cur[60*10:] = [0]*(70*10)  # cur is zero

                cur_cycle.extend(cur)
                vol_cycle.extend(row["plus_vol"])

                if expected == 4:  # 完成一轮
                    result[cell_id]["plus_cur"].append(cur_cycle)
                    result[cell_id]["plus_vol"].append(vol_cycle)
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
        end = 0.1 * (len(result[cell_id]["plus_vol"][0]) - 1)  # 计算终点值：519.9
        arr = np.linspace(0, end, num=len(result[cell_id]["plus_vol"][0]))  # 关键函数[1,5](@ref)
        result[cell_id]["time"] = arr
    return result   
       

def run(df_origin, picture_save_path, pickle_save_path):
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

                           df_cleaned --[用于算法模块计算的传入数据]--
                           columns 说明:
                               
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
        # --------data clean ---------#
        # 1. 判断cvs文件是否符合要求，即算法执行必须数据列名存在
        # todo: 增加不同文件数据的分别判断
        check_column = ['cell_id', 'plus_repeat_cnt', 'plus_present_cnt', 'plus_cur', 'plus_vol']
        rlt = check_invalid_col(df_origin, check_column)
        if not rlt:
            rlt_res['ErrorCode'][0] = -1
            rlt_res['ErrorCode'][2] = "csv文件列名不满足算法计算要求，缺少关键列名"
            return rlt_res, []
        
        # 2. 数据清洗
        rlt_res['ErrorCode'][0], df_cleaned = data_clean(df_origin)
        # 3. 对脉冲检测数据进行整合
        all_plus_data = merge_plus_data(df_cleaned)
        add_out_dir_info(rlt_res, '脉冲数据', all_plus_data, '', '')
        return rlt_res, df_cleaned
    
    except Exception as e:
        rlt_res['ErrorCode'][0] = -99
        rlt_res['ErrorCode'][2] = f'数据清洗报错{traceback.print_exc()}'
        return rlt_res, []
