# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import pickle
from .abusive_lib import Abusive_condition
import traceback
from tool_utils.pvlog import Logger

level = "error"
log = Logger('./logs/abusive_condition.log', level=level)

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


def run(df_raw, battery_type, full_cap, picture_save_path, pickle_save_path, **kwargs):
    try:
        rlt_res = {
        "code_id": 3,
        "describe": "abusive_condition",
        "out": {},
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }
    
        t1 = time.time()
        df = df_raw.copy()
        df.index = range(df.shape[0])

        if battery_type == 'NCM':
            vol_ub = 4.35
            vol_lb = 2.5
            temp_ub = 65
            temp_lb = -20
        elif battery_type == 'LFP':
            vol_ub = 3.80
            vol_lb = 2.0
            temp_ub = 65
            temp_lb = 0
        
        summary= []
        advice= []
        # 增加数据上传正确判断， 上传正确才评估充电工控
        if df['vol_max'].max() <= df['vol_max'].mode()[0] :
            # summary.append(f'数据不支持，充电工况暂无法评估。')
            add_out_dir_info(rlt_res, '电池组充电工况评分', 'N/A', summary, [])
            add_out_dir_info(rlt_res, '电池组充电工况数据', ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'], '', '')
            return rlt_res  
        
        # 增加数据上传正确判断， 上传正确才评估充电工控
        if kwargs['vol_max'] != '/':
            if battery_type == 'NCM' and kwargs['vol_max'] >4:
                vol_ub = kwargs['vol_max']
            elif battery_type == 'LFP' and kwargs['vol_max'] >3.5:
                vol_ub = kwargs['vol_max']

        if kwargs['temp_max'] != '/' and kwargs['temp_max'] > 50:
            temp_ub = kwargs['temp_max']

        """ 将数据切分成 1 个时段 """
        table_rows = 1
        interval = df.shape[0] // table_rows
        remainder = df.shape[0] % table_rows

        table = [['时间段', '过充(%)', '过放(%)', '过温(%)', '低温(%)', '高倍率充电(%)']]
        start = 0
        period_ratio = []
        period_start = []
        period_end = []

        for i in range(table_rows):
            if remainder != 0 and i == 0:
                end = start + interval + remainder
            else:
                end = start + interval

            period = df.loc[start:end]
            period_start.append(period['date_time'].values[0])    
            period_end.append(period['date_time'].values[-1])            

            charge_times = period['time_diff_seconds'].sum()
            # 判断过充
            filtered_period = period[(period.filter(like='vol_max') > (vol_ub+0.05)).any(axis=1)]
                 
            if charge_times > 0 and len(filtered_period) > 0:
                over_charge_times = filtered_period['time_diff_seconds'].sum()
                over_charge = round((over_charge_times*100 / charge_times),3)
                max_vol = filtered_period['vol_max'].max()
                diff_vol = round(max_vol - vol_ub,3)
                if diff_vol < 0.05: 
                    over_charge_msg = '轻度'
                elif diff_vol < 0.1:
                    over_charge_msg = '中度'
                else:
                    over_charge_msg = '重度'
            else:
                over_charge = 0
            

            filtered_period= period[((period.filter(like='temp_max') >temp_ub).any(axis=1))]

            if charge_times > 0 and len(filtered_period) > 0:  
                high_temp_times = filtered_period['time_diff_seconds'].sum()
                high_temp = round((high_temp_times*100 / charge_times),3)
                max_temp = filtered_period['temp_max'].max()
                diff_temp = round(max_temp - temp_ub,3)
                if diff_temp < 5: 
                    over_temp_msg = '轻度'
                elif diff_temp < 10:
                    over_temp_msg = '中度'
                else:
                    over_temp_msg = '重度'
            else:   
                high_temp = 0
           

            filtered_period= period[((period.filter(like='temp_min') < temp_lb).any(axis=1))]

            if charge_times > 0 and len(filtered_period) > 0:  
                low_temp = round((low_temp_times*100 / charge_times),3)
                low_temp_times = filtered_period['time_diff_seconds'].sum()
                min_temp = filtered_period['temp_min'].max()
                diff_low_temp = round(temp_lb - min_temp ,3)
                if diff_low_temp < 5: 
                    low_temp_msg = '轻度'
                elif diff_low_temp < 10:
                    low_temp_msg = '中度'
                else:
                    low_temp_msg = '重度'
            else:   
                low_temp = 0

            if full_cap != '/' and full_cap > 1:
                filtered_period= period[(period['current'] > full_cap*2)]
                if charge_times > 0 and len(filtered_period) > 0:  
                    high_rate_times = filtered_period['time_diff_seconds'].sum()
                    high_rate = round((high_rate_times*100 / charge_times),3)    
                else:           
                    high_rate = 0
            else:
                high_rate = 0
          
            # 分别判断gular_rate, high_temp, low_temp, over_charge, over_discharge
            need_advice = False
            if over_charge > 0:
                # summary.append(f'充电期间最高单体电压{max_vol}V，超过系统保护值{vol_ub}V，过充时长占比{over_charge}%, 属于{over_charge_msg}过充。')
                # summary.append(f'充电期间最高单体电压{max_vol}V，超过系统保护值{vol_ub}V，过充时长占比{over_charge}%， 该占比越高，代表电池管理系统未及时执行相对应保护功能。')
                summary.append(f'充电期间最高单体电压{max_vol}V，超过系统保护值{vol_ub}V，持续时间占比{over_charge}%')
                #if over_charge_msg == '重度' or over_charge_msg == '中度':
                    # advice.append(f'充电期间最高单体电压{max_vol}V，超过保护值{vol_ub}V，差值为{diff_vol}V, 属于{over_charge_msg}过充，疑似车载管理BMS系统存在问题，建议检查。')
                # advice.append(f'电池组出现充电过充，电池过充存在爆炸风险，建议原厂检查BMS系统可靠性。')
                # advice.append(f'充电期间出现最高单体电压超过BMS允许最高单体电压。')
                # advice.append(f'单体电压过高')
                need_advice = True

            if high_temp > 0:
                # summary.append(f'充电期间最高温度{max_temp}℃，超过系统保护值{temp_ub}℃，过温时长占比{high_temp}%，该占比越高，代表电池管理系统未及时执行相对应保护功能。')
                summary.append(f'充电期间最高温度{max_temp}℃，超过系统保护值{temp_ub}℃，持续时间占比{high_temp}%')
                #if over_temp_msg == '重度' or over_temp_msg == '中度':
                    #advice.append(f'充电期间最高温度值{max_temp}℃，超过系统保护值{temp_ub}℃，差值为{diff_temp}℃, 属于{over_temp_msg}过温，疑似车载管理BMS系统存在问题，建议检查。')
                #advice.append(f'电池组出现充电过温，电池过温存在热失控风险，建议原厂检查BMS系统可靠性。')
                # advice.append(f'温度过高')
                need_advice = True
            
            if low_temp > 0:    
                summary.append(f'充电期间最低温度{min_temp}℃，低于该电池类型平均运行最低充电保护温度值{temp_lb}℃，持续时间占比{low_temp}%')
                #if low_temp_msg == '重度' or low_temp_msg == '中度':
                    # advice.append(f'充电期间最低温度值{min_temp}℃，低于系统保护值{temp_lb}℃，差值为{diff_low_temp}℃, 属于{low_temp_msg}过低温，疑似车载管理BMS系统存在问题，建议检查。')
                # advice.append(f'电池组出现低温充电，低温充电存在安全风险，建议原厂检查BMS系统可靠性。')
                # advice.append(f'低温充电现象')
                need_advice = True

            if high_rate > 0:
                summary.append(f'超过2C的高倍率充电时长占比{high_rate}%，高倍率充电极大缩短充电时长，但需注意不要过于频繁')
                #if high_rate > 90:
                #    advice.append(f'超过1.5C的高倍率充电时长占比{high_rate}%，高倍率充电时间长会增加电池老化或电池鼓包现象概率。')

            total_ratio = over_charge  + high_temp + low_temp + high_rate *0.2  # give less weight to high_rate
            if total_ratio == 0:
                summary.append(f'充电过程合格，无电池过热、电压异常、低温充电等异常情况')
            
            if need_advice:
                advice.append(f'【BMS充电保护】')

            period_ratio.append(total_ratio)

            # 制作返回列表
            # length = len(period['date_time'][0])
            # table.append([period['date_time'].values[0] + '-' + period['date_time'].values[-1], over_charge,
            #               0, high_temp, low_temp, high_rate])

            # start_time_str = period['date_time'].values[0].strftime('%Y-%m-%d %H:%M:%S')
            # end_time_str = period['date_time'].values[-1].strftime('%Y-%m-%d %H:%M:%S')

            table.append([f'{period_start[i]}-{period_end[i]}', over_charge,
                        0, high_temp, low_temp, high_rate])
            start = end + 1

        # 新增数据异常值对评分的影像
        abnormal_value_ratio = kwargs['voltage_abnormal_value_ratio'] + kwargs['temp_abnormal_value_ratio'] + kwargs['soc_abnormal_value_ratio']
        
        score =  round(100 - np.mean(period_ratio)/2 - abnormal_value_ratio/3, 2)

        add_out_dir_info(rlt_res, '电池组充电工况评分', score, summary, advice)
        add_out_dir_info(rlt_res, '电池组充电工况数据', table[1], '', '')

        t2 = time.time()
        log.logger.info(f"Abusive condition time: {round(t2 - t1, 1)} seconds")

        return rlt_res
    except Exception as e:
        log.logger.error(f"Abusive condition fails: { traceback.print_exc()}")
        
        add_out_dir_info(rlt_res, '电池组充电工况评分', 'N/A', [], [])
        add_out_dir_info(rlt_res, '电池组充电工况数据', ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'], '', '')
        return rlt_res



