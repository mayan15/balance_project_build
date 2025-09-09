import pandas as pd
import numpy as np
import time
import traceback


def run_temp_sensor_check(df, battery_type, log): 
    """

    """
    try:
        result_dict = {}
        st = time.time()
        temp_max_colum = df['temp_max']
        temp_min_colum = df['temp_min']
        # temp_gun = df['charge_gun_temp']   

        summary = []
        advice = []
        ''' 计算温度列最大值和最小值，若一样，说明温度采集或者上传有问题'''
        max_temp_max = temp_max_colum.max()
        min_temp_max = temp_max_colum.min()

        max_temp_min = temp_min_colum.max()
        min_temp_min = temp_min_colum.min()

        temp_correlation_score = 'N/A'
        
        # 新增：若某一列数据无变化时，不进行相关性计算 或者最大值小于最小值也不计算
        if max_temp_max == min_temp_max  or max_temp_min == min_temp_min or max_temp_max < max_temp_min:
            temp_sensor_value_correct_score = 'N/A'   # 数据未正确上传时，只需要上传分数，不需要上传具体的建议
            temp_correlation_score = 'N/A' 
            summary.append(f'数据不支持，BMS温度传感器性能暂无法评估。')
        else:
            # 温度传感器有效性 1. 计算温度间相关性   2.计算BMS温度与枪温的相关性(暂时先保留，后续根据实际情况调整)
            correlation_temp_by_self = np.corrcoef(temp_max_colum, temp_min_colum)[0, 1]
            # correlation_temp_by_gun = np.corrcoef(temp_max_colum, temp_gun)[0, 1]
            if correlation_temp_by_self > 0.2: 
                temp_sensor_value_correct_score = round(100*correlation_temp_by_self,2)
            else:
                # temp_sensor_value_correct_score = 10
                temp_sensor_value_correct_score = 55 
                
            ''' 计算温度传感器有效性判断和建议 '''
            if temp_sensor_value_correct_score < 20:
                # 充电过程中电芯间温度传感器测量有效性系数
                summary.append(f'BMS温度传感器有效性指标值{round(temp_sensor_value_correct_score/100,2)}，低于该类型车辆BMS温度传感器测量有效性平均值， 该值越低，表明部分温度感器存在测量误差或反应不灵敏问题， 具体以4S店检测为准。')
                # advice.append('BMS温度传感器存在较严重的测量响应问题，建议检查BMS温度传感系统是否工作正常。')
                # advice.append(f'电池包温度传感器测量有效性指标{round(temp_sensor_value_correct_score/100,2)}，正常范围0.2-1.0，疑似存在BMS部分温度探头值检测失效，建议检查车载BMS系统。')
                # advice.append(f'电池包温度传感器有效性值指标值{temp_sensor_value_correct_score}，正常范围60-100，疑似存在BMS温度检测失效问题，概率1%，疑似存在充电数据未正确上报，概率99%。')
                advice.append(f'电池组温度传感器测量误差')
            else:
                # summary.append(f'电池包温度传感器有效性指标值{round(temp_sensor_value_correct_score/100,2)}，数值正常。')
                summary.append(f'BMS温度传感器有效性指标值{round(temp_sensor_value_correct_score/100,2)}，温度传感器工作正常。')
           
        result_dict['error'] = 0
        result_dict['class'] = 'sensor'
        result_dict['name'] = 'temp'
        
        result_dict['score'] = [temp_sensor_value_correct_score, temp_correlation_score]
        result_dict['summary'] = summary
        result_dict['advice'] = advice
        
        log.logger.debug(f"temp sensor check calculate time: {round(time.time()-st,2)} seconds")
        return result_dict
    except Exception as e:
        log.logger.error(f"temp sensor check error: {traceback.print_exc()}")
        result_dict['score'] = ['N/A', 'N/A']
        result_dict['summary'] = []#['数据不足，无法进行BMS温度传感器分析。']
        result_dict['advice'] = [] #['数据异常，无法进行BMS温度传感器分析']
        result_dict['error'] = 99
        return result_dict

