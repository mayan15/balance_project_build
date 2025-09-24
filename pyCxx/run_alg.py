import os
import pandas as pd
import json
from datetime import datetime

# import shutil
# import traceback
# from kafka import KafkaConsumer
# import configparser

# from tool_utils.kafka_to_monitor import send_report_msg
# from tool_utils.generate_report import generate_report
# from tool.pvlog import Logger
from tool_utils.pvlog import Logger

from pyCxx.data_clean_and_overview import data_clean_overview
from pyCxx.generate_json import generate_json

# from pyCxx.generate_json import charge_pile_json     
# from pyCxx.SOH_electro_chemistry import SOH_electro          # 电化学   
# from pyCxx.abusive_condition import abusive_condition         # 滥用工况
# from pyCxx.consistency import consistency     # 一致性检查
# from pyCxx.sensor_check import sensor_check  # 传感器异常检测   
# from pyCxx.safe_judge import safe_judge  # 充电安全性评估

from pyCxx.consistency import consist_balance_data   # 基于均衡数据的一致性检查
from pyCxx.SOH_rc import first_order    #  
from pyCxx.SOH_ocv import cal_soh_autocap  

level = 'info'
log = Logger('./logs/run_alg.log', level=level)

# 存放所有算法模块的执行结果， 最终用于报告生成
def alg_execution(df_pulse_data, dict_autocap_data, dict_balance_data, cell_config_msg, filename):
    alg_rlt_list = []
    current_date = datetime.now().strftime("%Y%m%d")     # 获取当前日期
    log.logger.info("开始执行算法模块...")
        
    # 构建报告存储文件夹路径
    first_level_folder = os.path.join('./report', current_date)
    report_folder_path = os.path.join(first_level_folder, filename)
    # 创建第一级文件夹
    os.makedirs(report_folder_path, exist_ok=True)
    data_clean_rlt = data_clean_overview.run(df_pulse_data, dict_autocap_data, dict_balance_data, cell_config_msg)
    alg_rlt_list.append(data_clean_rlt)
    if data_clean_rlt['ErrorCode'][0] == 0:
        # 1.内阻检测算法
        pulse_rlt = first_order.run(data_clean_rlt['out']['pulse_data'][0], data_clean_rlt)
        alg_rlt_list.append(pulse_rlt)
        # 2.todo 特定工况容量计算
        soh_info = cal_soh_autocap.run(data_clean_rlt['out']['autocap_data'][0], data_clean_rlt)
        alg_rlt_list.append(soh_info)
        # 3. 基于均衡数据的一致性检查
        consist_rlt = consist_balance_data.run(data_clean_rlt['out']['balance_data'][0], data_clean_rlt)
        alg_rlt_list.append(consist_rlt)
        # 4.生成json报告文件
        rlt_json = generate_json.run(data_clean_rlt, alg_rlt_list, report_folder_path)
        return rlt_json
    else:
        log.logger.error(data_clean_rlt['ErrorCode'][0], data_clean_rlt['ErrorCode'][2])
        return data_clean_rlt
    
def check_folder():
    try:
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        if not os.path.exists('./csv_origin'):
            os.mkdir('./csv_origin')
        if not os.path.exists('./csv_history'):
            os.mkdir('./csv_history')
        if not os.path.exists('./csv_error'):
            os.mkdir('./csv_error')
        if not os.path.exists('./report'):
            os.mkdir('./report')
        if not os.path.exists('./report_pile'):
            os.mkdir('./report_pile')
    except Exception as e:
        log.logger.error("自动化生成文件夹报错，请检查代码运行权限!")
