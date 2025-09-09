
import traceback
import pickle
import numpy as np
import pandas as pd
import time


def run_temp_consis(df, log, max_temp=0, min_temp=0,data_cleaned=None):
    try:
        result_dict = {}
        st = time.time()

        summary =[]
        advice =[]
        ''' 计算充电结束后 两个温度列产生的最大温度差'''
        diff = df['temp_max'] - df['temp_min']

        
        temp_dec_score = 'N/A'
        temp_cv_score = 'N/A'

        max_temp_max = df['temp_max'].max()
        min_temp_max = df['temp_max'].min()

        max_temp_min = df['temp_min'].max()
        min_temp_min = df['temp_min'].min()

        diff_max =  max_temp_max - df['temp_max'].iloc[1]   # 温升

        if (max_temp_max == min_temp_max  or  max_temp_min == min_temp_min ) :
            temp_dec_score = 'N/A'
            temp_cv_score = 'N/A'
            # summary.append(f'数据不支持，温度一致性和热状态暂无法评估。')
        else: 
            if diff.max() > diff.min() and diff.max() > 0:
                temp_dec = diff.max()     
                if temp_dec <= 15:
                    temp_dec_score = 100 - temp_dec   # 根据温度差值来进行评分
                #elif temp_dec < 25:
                    # temp_dec_score = round(100/(temp_dec/15),2)    # 温度差评分: 25℃对应基准60分
                #    temp_dec_score = 85 - temp_dec
                else:
                    temp_dec_score = 74 - temp_dec    # 温度差评分: 每比 25℃ 高 1 ℃，减去1分

                # 2025.2.14 增加对最高温度/最低温度节号分布的判断，并再报告中增加对特定温度节号排查的建议
                # need_check_temp_no = []
                # try:
                #     rlt_no_describe = list(data_cleaned['out']['max_temp_no_describe'][0].keys())
                #     rlt_no_precent =  list(data_cleaned['out']['max_temp_no_describe'][0].values())     
                #     if  rlt_no_describe[0] > 0 and rlt_no_precent >70:
                #         need_check_temp_no.append(rlt_no_describe[0])

                #     need_check_temp_no.append(df['max_temp_no_describe'].iloc[-1])
                    
                #     rlt_no_describe = list(data_cleaned['out']['min_temp_no_describe'][0].keys())
                #     rlt_no_precent =  list(data_cleaned['out']['min_temp_no_describe'][0].values())     
                #     if  rlt_no_describe[0] > 0 and rlt_no_precent >70:
                #         need_check_temp_no.append(rlt_no_describe[0])

                #     need_check_temp_no.append(df['min_temp_no_describe'].iloc[-1])
                #     need_check_temp_no = list(set(x for x in need_check_temp_no if x != 0))       
                # except   Exception as e:
                #     log.logger.error(f"temp consis error: {traceback.print_exc()}")

                need_advice = False   
                ''' 计算温度一致性相对指标判断和建议 '''
                if temp_dec_score < 60:
                    # summary.append(f'充电过程中温度极差值为{temp_dec}℃，超出该类型车型平均值{over_nomal_percent}℃，温度极差越大，电池组温度一致性越差。')
                    # summary.append(f'充电过程中温度极差值为{temp_dec}℃，超出该类型车型平均值，温度极差越大，电池组温度一致性越差。')
                    summary.append(f'最大温度差{temp_dec}℃，数值超出正常（<15℃）范围')
                    # advice.append(f'温差异常')
                    need_advice = True 
                    # summary.append(f'电池组最大温度差为{temp_dec}℃，数值高。') 
                    # if len(need_check_temp_no) > 0:
                    #     advice.append(f'电池组最大温度差为{temp_dec}℃，正常范围1-25℃，疑似存在局部散热问题，建议检查BMS温度探头{need_check_temp_no}检测是否准确。')
                    # else:
                    #     advice.append(f'电池组最大温度差为{temp_dec}℃，正常范围1-25℃，疑似存在局部散热问题，建议检查BMS温度检测是否准确。')
                else:
                    # summary.append(f'电池组温度极差值{temp_dec}℃，温度一致性正常。')
                    summary.append(f'最大温度差{temp_dec}℃，温度一致性合格')

                '''计算温升 （热状态）'''   
                if diff_max >= 0 :
                    if diff_max <=30:
                        temp_cv_score = 100 - diff_max 
                    elif diff_max < 40:
                        temp_cv_score = 89 - diff_max 
                    else:
                        temp_cv_score = 30

                # elif diff_max < 30:
                #     temp_cv_score = round(100/(diff_max/15),2)    # 温度差评分: 30℃对应基准60分
                # else:
                #     temp_cv_score = 90 - diff_max    # 温度差评分: 每比 30℃ 高 1 ℃，减去1分
                
                if temp_cv_score != 'N/A':
                    if temp_cv_score < 60:
                        # summary.append(f'充电过程中温度极差值为{temp_dec}℃，超出该类型车型平均值{over_nomal_percent}℃，温度极差越大，电池组温度一致性越差。')
                        summary.append(f'最大温升为{diff_max}℃，数值超出正常（<30℃）范围')
                        need_advice = True
                    else:
                        summary.append(f'最大温升{diff_max}℃，温升合格')
                 
                if need_advice:
                    advice.append(f"【电池组散热系统】") 
                # if temp_cv_score <= 36:
                #     summary.append(f'电池组电芯温度离散性指标值为{std}，数值高。')
                #     #advice.append(f'电池组电芯温度离散性指标值为{std}，正常范围0-2，疑似存在局部散热问题，建议检查电池组散热系统。')
                # elif temp_cv_score< 44:
                #     summary.append(f'电池组电芯温度离散性指标值为{std}，数值偏高。')
                # else:
                #     summary.append(f'电池组电芯温度离散性指标值为{std}，数值正常。')
            else:
                temp_dec_score = 'N/A'
                temp_cv_score = 'N/A'
                # summary.append(f'数据不支持，温度一致性和热状态暂无法评估。') 
        
        if temp_dec_score != 'N/A' and temp_cv_score!= 'N/A':
            temp_score = round(temp_dec_score*0.6 + temp_cv_score*0.4,2)
        else:
           temp_score = temp_dec_score

        result_dict['error'] = 0
        result_dict['class'] = 'consistency'
        result_dict['name'] = 'temperature'
        result_dict['score'] = [temp_score, temp_cv_score]
        result_dict['summary'] = summary
        result_dict['advice'] = advice
        log.logger.debug(f"Consist temp calculate time: {round(time.time()-st,2)} seconds")
        return result_dict
        
    except Exception as e:
        log.logger.error(f"temp consis error: {traceback.print_exc()}")
        result_dict['error'] = 99
        result_dict['score'] = ['N/A', 'N/A']
        result_dict['summary'] = []#['数据不足，温度一致性暂无法评估']
        result_dict['advice'] = []#['数据异常，温度一致性相关暂无建议']
        return result_dict
        


