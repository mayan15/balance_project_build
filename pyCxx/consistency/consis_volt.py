
import traceback 
import numpy as np
import pandas as pd
import time
import pickle
from tool_utils.pvlog import Logger

def run_volt_consis(df, log, max_volt=0, min_volt=0, soh_rlt=None, data_cleaned=None):
    """
    该函数用于计算电池电压一致性，并生成一致性报告。其主要步骤包括：
    1. 计算电池电压的均值、标准差及其差异，评估电池一致性。
    2. 针对每个电池包，调用 Difovstd 类进行电压一致性分析。
    3. 根据计算的分数、最差电池和最佳电池生成一致性报告。
    4. 最后将计算结果保存为一个 pickle 文件并记录日志。

    参数:
    df (pd.DataFrame): 输入的数据框，包含不同电池包的电压数据。
    all_pack_volts (list): 包含所有电池包名称的列表，供 Difovstd 进行电压分析。
    result_save_path (str): 结果保存的文件路径，最终会将一致性计算结果保存为 pickle 文件。
    log (logger): 日志记录对象，用于输出计算过程中的日志信息。

    返回:

    """
    try:
        result_dict = {}
        st = time.time()
        summary =[]
        advice =[]
        limit_dec = 0.5
        vol_dec = 0
        soc_end = data_cleaned['out']['结束SOC'][0]
        ''' 计算充电结束后 最大值与最小值电压差''' 
        if min_volt != '/'  and min_volt > 0.01 and df['vol_max'].max() > df['vol_max'].mode()[0] and max_volt > min_volt:
            vol_dec = round(max_volt - min_volt,3)
            # 分三元和铁锂进行判断， 且铁锂分SOC进行判断   
            # 三元： 0.2V
            if data_cleaned['out']['电池类型'][0] == "NCM":  # 三元
                limit_dec = 0.2
                if vol_dec < 0.2:
                    vol_dec_score = round(100- vol_dec*99)
                # elif vol_dec < 0.5:
                #     vol_dec_score = round(100- vol_dec*80,2)    # 电压差评分: 0.5V对应基准60分
                elif vol_dec < 0.7:
                    vol_dec_score = round(50 - (vol_dec-0.2)*100,2)    # 电压差评分: 每比0.2高0.1V，再减去10
                else:
                    vol_dec_score = 10  
            # 铁锂: 0.35
            elif data_cleaned['out']['电池类型'][0] == "LFP":  # 铁锂
                
                if soc_end >= 64 and soc_end <= 76:
                    limit_dec_by_soc = 0.15
                elif soc_end == 99 or soc_end == 100: 
                    limit_dec_by_soc = 0.45
                else:
                    limit_dec_by_soc = 0.25

                limit_dec = limit_dec_by_soc

                if vol_dec < limit_dec_by_soc:
                    vol_dec_score = round(100- vol_dec/limit_dec_by_soc*40)
                # elif vol_dec < 0.5:
                #     vol_dec_score = round(100- vol_dec*80,2)    # 电压差评分: 0.5V对应基准60分
                elif vol_dec < 0.85:
                    vol_dec_score = round(50 - (vol_dec-limit_dec_by_soc)*100,2)    # 电压差评分: 每比0.2高0.1V，再减去10
                else:
                    vol_dec_score = 10

            if vol_dec_score < 60:  
                # over_nomal_percent = round(vol_dec/limit_dec,2)
                #if over_nomal_percent > 2:
                # summary.append(f'充电结束电压极差值为{vol_dec}V，超出该电池类型平均值{over_nomal_percent}倍，电压极差值越大，电池组电压一致性越差。')
                # summary.append(f'充电结束SOC值为：{soc_end}%，单体电压最大差值为{vol_dec}V，超出该电池类型正常范围，电压极差值越大，电池组电压一致性越差。')
                summary.append(f'电压差{vol_dec}V，数值超出正常（<{limit_dec}V)范围')
                advice.append(f'【电池组电压一致性】')
                #else:
                #    summary.append(f'电池组电压极差值为{vol_dec}V，超出正常阈值{over_nomal_percent}倍，存在电压一致性问题。')
                #    advice.append(f'电池组电压一致性差，建议返回原厂进行均衡维护或更换。')
            else:
                # 增加结束SOC判断，以及电池可充容量综合判断，是否提醒客户以充满时一致性检测为准： 
                summary.append(f'电压差{vol_dec}V，电压一致性合格')

            ''' 计算充电过程中，平均值与最大值序列值压差 '''
            diff =  df['vol_max'][1:] - df['vol_mid'][1:]
            std  = round(diff.std(),4)
            max_v  = diff.max()
            min_v  = diff.min()
            std_std = abs(max_v-min_v)/12**0.5
            limit_h = round(std_std*2.65,4)
            limit_l = round(std_std*0.35,4)

            if (std < limit_l ) :
                vol_mid_score = 98
            elif (std > limit_h ):
                vol_mid_score = 59
            else:
                vol_mid_score = 38* (std-limit_h)/(limit_l-limit_h) + 60
 
        # 数据未正常上传: 最大值上传不正确，未上传最小值 
        else:
            vol_dec_score = 'N/A'
            vol_mid_score = 'N/A'
            # if min_volt == '/' or min_volt < 0.01:
            #     summary.append(f'单体电压最小值未上传，电压一致性暂无法评估。')
            # else:
            #     summary.append(f'单体电压最大值未上传，电压一致性暂无法评估。')
                
        result_dict['error'] = 0
        result_dict['class'] = 'consistency'
        result_dict['name'] = 'voltage'
        result_dict['score'] = [vol_dec_score, vol_mid_score, vol_dec]
        result_dict['summary'] = summary
        result_dict['advice'] = advice
        
        log.logger.debug(f"Consist Volt calculate time: {round(time.time()-st,2)} seconds")
        return result_dict
    except Exception as e:
        log.logger.error(f"volt consis error: {traceback.print_exc()}")
        result_dict['error'] = -99
        result_dict['score'] = ['N/A', 'N/A']
        result_dict['summary'] = []#['数据不足，电压一致性暂无法评估。']
        result_dict['advice'] = []
        return result_dict

