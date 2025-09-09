import pandas as pd
import numpy as np
import time
import traceback


def sigmoid_normalize_x(x, mu=1, s=5):
    return 1 / (1 + np.exp(-(x - mu) / s))

def run_cur_sensor_check(df, battery_type, log):
    """

    """
    try:
        result_dict = {}
        st = time.time()
        # ''' 计算电流传感器有效性判断和建议 '''
        summary = []
        advice = []
        cur_sensor_value_correct_score = 'N/A' 
        # 计算电流传感器有效性判断和建议  (只判断电流传感器有效性，根据总电流来判断)
        # 计算BMS总电流与充电桩总电流关系（两者理论是强相关）
        correlation_cur_max = round(np.corrcoef(df['current'], df['charge_out_cur'])[0, 1],3)
        cur_correlation_score = round(100* correlation_cur_max,2)
        # 计算BMS总电流与充电桩总电流的绝对误差和相对误差
        # cur_rule = 0.16* df['current'].mean()
        # cur_rule = round(0.16* df['charge_out_cur'].mean(),2)
        cur_rule = 25
        cur_total_error = round(abs(df['current']- df['charge_out_cur']).mean(),2)
        if cur_total_error > 50:
            cur_sensor_value_correct_score = 'N/A'
            cur_correlation_score = 'N/A'
            # summary.append(f'数据不支持，电流传感器测量有效性暂无法评估。')
        else:    
            cur_total_relative_error = round(cur_rule/cur_total_error ,2) 
            cur_total_relative_normalize = round(sigmoid_normalize_x(cur_total_relative_error),2)
            cur_sensor_value_correct_score = cur_total_relative_normalize*100 

            if cur_sensor_value_correct_score < 50:
                # 充电过程中电芯间电流传感器测量有效性系数 
                # summary.append(f'电流传感器测量有效性指标值{cur_total_relative_normalize}， 低于车辆BMS电流传感器测量有效性平均值{round(3/cur_total_relative_error,2)}倍，该值越低，表明电流传感器测量值与真实值偏差越大。')
                # summary.append(f'电流传感器测量有效性指标值{cur_total_relative_normalize}，该值越低，表明电流传感器测量值与真实值偏差越大。')
                summary.append(f'BMS上传总电流数值与充电桩总电流数值存在偏差，平均相差{cur_total_error}A，超出正常（<{cur_rule}A）范围')
                # advice.append(f'BMS电流传感器测量有效性指标值{round(cur_sensor_value_correct_score/100,2)}，正常范围0.6-0.1，疑似存在BMS电流检测误差问题，建议检查车载BMS系统。')
                advice.append(f'【BMS总电流测量数值】')
            # elif vol_sensor_value_correct_score < 70:
            #     # 充电过程中电芯间电压传感器测量有效性系数
            #     summary.append(f'电池包电压传感器有效性指标值{vol_sensor_value_correct_score}，数值偏低。')
                #advice.append(f'电池包电压传感器测量有效性指标值{vol_sensor_value_correct_score}，正常80及以上，可能存在BMS电压检测值误差大问题，建议检查车载BMS系统或数值上传。')
            else:
                summary.append(f'总电流测量数值合格')

        # if cur_correlation_score < 60:
        #     summary.append(f'BMS电流传感器测量灵敏性指标值{round(cur_correlation_score/100,2)}，数值低。')
        #     if cur_sensor_value_correct_score > 60:  # 充电过程中电芯间电流传感器测量有效才看相关性问题
        #         advice.append(f'BMS电流传感器测量灵敏性指标值{round(cur_correlation_score/100,2)}，正常范围0.6-0.1，疑似存在BMS电流检测误差问题，建议检查车载BMS系统。')
        #     # advice.append('BMS电压传感器存在较严重的测量线
        result_dict['error'] = 0
        result_dict['class'] = 'sensor'
        result_dict['name'] = 'cur_sensor_check'
        result_dict['score'] = [cur_sensor_value_correct_score,cur_correlation_score]
        result_dict['summary'] = summary
        result_dict['advice'] = advice
    
        # log.logger.debug(f"current sensor check calculate time: {round(time.time()-st,2)} seconds")
        return result_dict
    except Exception as e:
        log.logger.error(f"current sensor check error: {traceback.print_exc()}")
        result_dict['error'] = 99
        result_dict['score'] = ['N/A', 'N/A']
        result_dict['summary'] = []#['数据不足，无法进行BMS电流传感器分析。']
        result_dict['advice'] = []
        return result_dict