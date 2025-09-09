import traceback 
import numpy as np
import pandas as pd
import time


def run_soc_consis(df, log, max_volt=0, min_volt=0, soh_rlt= None, data_cleaned=None):
    """
    该函数用于计算电池SOC一致性，并生成一致性报告。其主要步骤包括：

    """
    try:
        result_dict = {}
        st = time.time()
        summary =[]
        advice =[]
        soc_chg_rate_score = 'N/A'
         
        # 电压值上传正确的基础上才计算SOC显示是否准确
        # if min_volt != '/' and df['vol_max'].max() >df['vol_max'].min() and data_cleaned['out']['起始SOC'][0] <= 50 and data_cleaned['out']['结束SOC'][0] > 96 and data_cleaned['out']['额定容量'][0] < 600:
        # if data_cleaned['out']['起始SOC'][0] <= 50 and data_cleaned['out']['结束SOC'][0] > 70 and data_cleaned['out']['额定容量'][0] < 600 and soh_rlt['out']['动力电池组剩余容量'][0] != 'N/A':
        if  data_cleaned['out']['额定容量'][0] < 600 and soh_rlt['out']['动力电池组剩余容量'][0] != 'N/A':
            # 计算充电过程中，SOC是否存在修正行为
            full_cap = float(data_cleaned['out']['额定容量'][0])
            soc_list = [50,60,70,80,90,99]
            filtered_df = df[df['soc'].isin(soc_list)]

            result_df = filtered_df.drop_duplicates(subset='soc', keep='last')
            bms_avg_soc = result_df['bms_ah'].diff().dropna()
            bms_avg_dec = bms_avg_soc.max() - bms_avg_soc.min()
            
            dec = round(bms_avg_dec/full_cap*100,2)
            
            #summary.append(f'SoC充电末端修正值为{dec}%')
            # if dec >15:
            #    advice.append(f'电池组SoC显示偏差。') 
                
            # 计算充电过程中，安时积分累计得SOC
            # soc_chg_rate =  round(data_cleaned['out']['累计充电容量'][0]*100/data_cleaned['out']['额定容量'][0],0)
            # soc_bms = data_cleaned['out']['结束SOC'][0] - data_cleaned['out']['起始SOC'][0]
            # soc_dec = abs(soc_bms - soc_chg_rate)

            # 按照地标规定方法进行SOC计算
            soc_chg =  (data_cleaned['out']['结束SOC'][0]/100 - (data_cleaned['out']['累计充电容量'][0] - df['bms_ah'])/soh_rlt['out']['动力电池组剩余容量'][0])*100  
            # soc_chg =  (data_cleaned['out']['结束SOC'][0]/100 - (data_cleaned['out']['累计充电容量'][0] - df['bms_ah'])/full_cap)*100  
            soc_chg = soc_chg.fillna(df['soc'].iloc[0])
            soc_chg_dec = round((soc_chg - df['soc']).max(),2)

            # vol_dec = round(max_volt - min_volt,3)
            if soc_chg_dec > 15:
                # if data_cleaned['out']['电池类型'][0] == "NCM":
                #     limit_dec = 0.2
                # else:
                #     limit_dec = 0.35

                #if vol_dec > limit_dec:
                soc_chg_rate_score = 30
                summary.append(f'核算车辆SOC（电量百分比）与充电量累计SOC存在一致性偏差，最大偏差值为{soc_chg_dec}%， 数值超出正常（<{15}%）范围')
                advice.append(f'【电池组SOC（电量百分比）】') 
            else:
                if len(data_cleaned['out']['soc_diff'][0]) >0 and df['soc'].iloc[data_cleaned['out']['soc_diff'][0][0]] < 90:
                    soc_chg_rate_score = 45
                    soc1 = df['soc'].iloc[data_cleaned['out']['soc_diff'][0][0]-1]
                    soc2 = df['soc'].iloc[data_cleaned['out']['soc_diff'][0][0]]
                    diff = data_cleaned['out']['soc_diff'][1]
                    summary.append(f'车辆SOC（电量百分比）充电过程中存在跳变，从{soc1}%跳变至{soc2}%，跳变幅度为{diff}%，数值超出正常（<{5}%）范围')
                    # summary.append(f'车辆SOC（电量百分比）充电过程中存在跳变，从{df['soc'].iloc[data_cleaned['out']['soc_diff'][0][0]-2]}%跳变至{df['soc'].iloc[data_cleaned['out']['soc_diff'][0][0]-1]}%，跳变幅度为{data_cleaned['out']['soc_diff'][1]}%，数值超出正常（<5%）范围')
                    # summary.append(f'车辆SOC（电量百分比）充电过程中存在跳变，跳变幅度为{data_cleaned['out']['soc_diff'][1]}%，数值超出正常（<5%）范围')
                    advice.append(f'【电池组SOC（电量百分比）】') 
                else:
                    soc_chg_rate_score = 99
                    summary.append(f'SOC（电量百分比）在整个充电过程中的变化平稳，充电末端修正值为{dec}%，与实际充电量匹配合格')

        else:
            if len(data_cleaned['out']['soc_diff'][0]) >0 and df['soc'].iloc[data_cleaned['out']['soc_diff'][0][0]] < 90:
                    soc_chg_rate_score = 45
                    soc1 = df['soc'].iloc[data_cleaned['out']['soc_diff'][0][0]-1]
                    soc2 = df['soc'].iloc[data_cleaned['out']['soc_diff'][0][0]]
                    diff = data_cleaned['out']['soc_diff'][1]
                    summary.append(f'车辆SOC（电量百分比）存在跳变，从{soc1}%跳变至{soc2}%，跳变幅度为{diff}%，数值超出正常（<{5}%）范围')
                    # summary.append(f'车辆SOC（电量百分比）充电过程中存在跳变，跳变幅度为{data_cleaned['out']['soc_diff'][1]}%，数值超出正常（<5%）范围')
                    advice.append(f'【电池组SOC（电量百分比）】')  
            else:
                soc_chg_rate_score = 'N/A' 
            # summary.append(f'数据不支持，SoC一致性暂无法评估。') 

        # ''' 计算电压一致性相对指标判断和建议 '''
        result_dict['error'] = 0
        result_dict['class'] = 'consistency'
        result_dict['name'] = 'soc'
        result_dict['score'] = [soc_chg_rate_score]
        result_dict['summary'] = summary
        result_dict['advice'] = advice
        
        log.logger.debug(f"Consist soc calculate time: {round(time.time()-st,2)} seconds")
        return result_dict
    except Exception as e:
        log.logger.error(f"soc consis error: {traceback.print_exc()}")
        result_dict['error'] = 99
        result_dict['score'] = ['N/A', 'N/A']
        result_dict['summary'] = []#['数据不足，SOC一致性暂无法评估。']
        result_dict['advice'] = []#['数据异常，SOC 一致性相关暂无建议']
        return result_dict