# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import pickle
import traceback
from tool_utils.pvlog import Logger

level = "error"
log = Logger('./logs/safe_judge.log', level=level)

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


# 对充电工况的控制流程进行判断 
def run(df_raw, battery_type, abusive_rlt, picture_save_path, pickle_save_path, **kwargs):
    try:
        rlt_res = {
        "code_id": 8,
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
            vol_need_down_cur = 4.10
            temp_need_stop = -5
            temp_need_down = 60
        elif battery_type == 'LFP':
            vol_need_down_cur = 3.41
            temp_need_stop = 0
            temp_need_down = 60
        
        # charge_max_out_cur = 0
        # if kwargs['charge_max_out_cur'] > 0:
        #     charge_max_out_cur = kwargs['charge_max_out_cur']
        # 2024-12-17 修改为判断输出电流是否大于BMS需求电流
        diff_cur = df['current'][1:] - df['bms_need_cur'][1:]
        #if charge_max_out_cur > 0:
        cur_inconformity_socre = 100 - (diff_cur > 20).sum()/df.shape[0] * 100
        #else:
        #    charge_max_out_cur = 100 
        summary= []
        advice= []
        socre = 'N/A'
        # 温度数据正常的情况下，才执行工况判断
        if kwargs['temp_sorce'] != 'N/A':
            # 判断温度过高时，是否执行了降电流措施：截取温度值大于50的数据段，并获取对应的电流值段
            current_values_ht = df.loc[df['temp_max'] > temp_need_down, 'current']
            if current_values_ht.shape[0] == 0:
                current_ht_socre = 100
            else:
                current_ht_mid = current_values_ht.mean()
                if current_ht_mid < 30:
                    current_ht_socre = 100
                elif current_ht_mid < 90:
                    current_ht_socre = 100 - current_ht_mid
                else:
                    current_ht_socre = 10 
            # 判断温度过低时，是否执行了充电（）  
            current_values_lt = df.loc[df['temp_min'] < temp_need_stop, 'current']
            cur_ratio_lt = (current_values_lt > 10).sum()/df.shape[0] * 100
            cur_ratio_lt_socre = 100 - cur_ratio_lt
            
            
            ad_flag = False
            if current_ht_socre < 60:
                summary.append(f'最高温度大于{temp_need_down}摄氏度时，充电电流大于{round(100-current_ht_socre,2)}A')
                ad_flag = True
                
            if cur_ratio_lt_socre < 90:
                summary.append(f'最低温度小于{temp_need_stop}摄氏度时，充电电流持续大于10A，占比{cur_ratio_lt}%')
                ad_flag = True
                
            if ad_flag:
                advice.append(f'【BMS充电控制管理】')

            if cur_inconformity_socre > 99 and current_ht_socre >99 and cur_ratio_lt_socre >99:
                summary.append(f'BMS充电控制管理合格')
        
            socre = cur_inconformity_socre*0.1+ current_ht_socre*0.4+ cur_ratio_lt_socre*0.5
            socre =  round(socre , 2)
        else:
            socre = 'N/A'
            # summary.append(f'数据不支持，充电控制合规性暂无法评估。')          
           
        rlt_res['ErrorCode'][0] = 0
        rlt_res['ErrorCode'][1] = 0
        rlt_res['ErrorCode'][2] = '无异常'
        rlt_res['summary'].append(summary)
        add_out_dir_info(rlt_res, '电池组充电安全评分', socre, summary, advice)
        add_out_dir_info(rlt_res, '输出电流异常值占比', round(100 - cur_inconformity_socre,2), '', '')
        
        # --------out picture ---------#

        # --------out pickle ---------#
        # with open(pickle_save_path + '/abusive_condition.pkl', "wb") as f:
        #     pickle.dump(rlt_res, f)

        t2 = time.time()
        log.logger.info(f"Abusive condition time: {round(t2 - t1, 1)} seconds")

        return rlt_res
    
    except Exception as e:
        log.logger.error(f"safe judge fails: {traceback.print_exc()}")
        add_out_dir_info(rlt_res, '电池组充电安全评分', 'N/A', [], [])       #'数据不足，充电安全暂无建议'
        add_out_dir_info(rlt_res, '输出电流异常值占比', 'N/A', '', '')
        return rlt_res