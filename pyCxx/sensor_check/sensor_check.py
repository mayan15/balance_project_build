import pandas as pd
import numpy as np
import time
import traceback

from .temp_snsor import run_temp_sensor_check
from .voltage_snsor   import run_vol_sensor_check
from .current_snsor   import run_cur_sensor_check

from tool_utils.pvlog import Logger

level = "debug"
log = Logger('./logs/consistency.log', level=level)

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
    rlt_res['out'][key].append(value)
    rlt_res['out'][key].append(confidence)
    rlt_res['out'][key].append(explanation)


def run(df_raw, battery_type, picture_save_path, pickle_save_path):
    """
       # df_raw: 原始数据
       # picture_save_path: 图片保存路径
       # pickle_save_path: 结果保存路径
       # 返回值：
       # rlt_res: 结果字典
    """
    try:
        rlt_res = {
        "code_id": 2,
        "describe": "consistentcy",
        "out": {},
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }
        st = time.time()
        # temp_sensor_valid_rlt = run_temp_sensor_check(df_raw, battery_type, log)
        vol_sensor_valid_rlt = run_vol_sensor_check(df_raw,battery_type, log)
        current_sensor_valid_rlt = run_cur_sensor_check(df_raw,battery_type, log)

        # temp_sensor_check_socker = 'N/A'
        vol_sensor_check_socker = 'N/A'
        cur_sensor_check_socker = 'N/A'
        
        # if temp_sensor_valid_rlt['score'][0] != 'N/A':
        #     temp_sensor_check_socker = round(temp_sensor_valid_rlt['score'][0], 2)
        if vol_sensor_valid_rlt['score'][0] != 'N/A':
            vol_sensor_check_socker =  round(vol_sensor_valid_rlt['score'][0]*0.9 + vol_sensor_valid_rlt['score'][1]*0.1, 2)
        if current_sensor_valid_rlt['score'][0] != 'N/A':    
            cur_sensor_check_socker = round(current_sensor_valid_rlt['score'][0]*0.9 + current_sensor_valid_rlt['score'][1]*0.1, 2)

        # if  temp_sensor_valid_rlt['score'][0] != 'N/A' and vol_sensor_valid_rlt['score'][0] != 'N/A' and current_sensor_valid_rlt['score'][0] != 'N/A':
        #     sensor_socre = round(temp_sensor_check_socker*0.1 + vol_sensor_check_socker*0.5 + cur_sensor_check_socker*0.4,2)
        if vol_sensor_valid_rlt['score'][0] != 'N/A' and current_sensor_valid_rlt['score'][0] != 'N/A':
            sensor_socre = round(vol_sensor_check_socker*0.6 + cur_sensor_check_socker*0.4,2)
        elif vol_sensor_valid_rlt['score'][0] != 'N/A':
            sensor_socre = vol_sensor_check_socker
        elif current_sensor_valid_rlt['score'][0] != 'N/A':
            sensor_socre = cur_sensor_check_socker
        else:
            sensor_socre = 'N/A'
        
        # log.logger.debug(f"BMS温度探头有效性检查结果: {temp_sensor_valid_rlt}")
        # log.logger.debug(f"BMS电压探头有效性检查结果: {vol_sensor_valid_rlt}")       
        log.logger.debug(f"sensor_check time: {round(time.time()-st,2)} seconds")  
        # BMS传感器有效性总评结果
        add_out_dir_info(rlt_res, 'BMS传感器有效性总评分', sensor_socre, '', '')
        add_out_dir_info(rlt_res, 'BMS电压传感器有效性评分', vol_sensor_check_socker, vol_sensor_valid_rlt['summary'], vol_sensor_valid_rlt['advice'])
        # add_out_dir_info(rlt_res, 'BMS温度传感器有效性评分', temp_sensor_check_socker, temp_sensor_valid_rlt['summary'], temp_sensor_valid_rlt['advice'])
        add_out_dir_info(rlt_res, 'BMS电流传感器有效性评分', cur_sensor_check_socker, current_sensor_valid_rlt['summary'], current_sensor_valid_rlt['advice'])
        return rlt_res   
    except Exception as e:
        log.logger.error(f"consistency error: {traceback.print_exc()}")
        add_out_dir_info(rlt_res, 'BMS传感器有效性总评分', 'N/A', '', '')
        add_out_dir_info(rlt_res, 'BMS电压传感器有效性评分', vol_sensor_check_socker, vol_sensor_valid_rlt['summary'], vol_sensor_valid_rlt['advice'])
        # add_out_dir_info(rlt_res, 'BMS温度传感器有效性评分', temp_sensor_check_socker, temp_sensor_valid_rlt['summary'], temp_sensor_valid_rlt['advice'])
        add_out_dir_info(rlt_res, 'BMS电流传感器有效性评分', cur_sensor_check_socker, current_sensor_valid_rlt['summary'], current_sensor_valid_rlt['advice'])
        return rlt_res
    