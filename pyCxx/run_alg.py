import os
import pandas as pd
# from tool.pvlog import Logger
from tool_utils.pvlog import Logger

from datetime import datetime
# import shutil
# import traceback

from pyCxx.data_clean_and_overview import data_clean_overview

from pyCxx.generate_json import generate_json
from pyCxx.generate_json import charge_pile_json     

from pyCxx.SOH_electro_chemistry import SOH_electro          # 电化学   
from pyCxx.abusive_condition import abusive_condition         # 滥用工况
from pyCxx.consistency import consistency     # 一致性检查
from pyCxx.sensor_check import sensor_check  # 传感器异常检测   
from pyCxx.safe_judge import safe_judge  # 充电安全性评估


from pyCxx.SOH_rc import first_order    #  
from pyCxx.SOH_ocv import cal_soh_dsoc_dcap  
# from kafka import KafkaConsumer
# import configparser
import json
# from tool_utils.kafka_to_monitor import send_report_msg
# from tool_utils.generate_report import generate_report


level = 'info'
log = Logger('./logs/run_alg.log', level=level)

# 存放所有算法模块的执行结果， 最终用于报告生成
def alg_execution(autocap_data_raw, pulse_data_raw, subfolder, log):
    alg_rlt_list = []
    current_date = datetime.now().strftime("%Y%m%d")     # 获取当前日期
    log.logger.info("开始执行算法模块...")
    
    # 获取JSON 配置文件关于算法模块的配置信息
    # file = 'config.json'
    # config = {}
    # with open('config.json', 'r', encoding='utf-8') as file:
    #     config = json.load(file)
    # data_overview_config = config['data_overview']

    # for index, df_value in enumerate(csv_data_df):      
    # 构建文件夹路径
    first_level_folder = os.path.join('./report', current_date)
    report_folder_path = os.path.join(first_level_folder, subfolder)
    # 第三级文件夹
    third_level_folders = ['picture', 'pickle']
    # 创建第一级文件夹
    os.makedirs(report_folder_path, exist_ok=True)
    # 创建第二级文件夹
    picture_folder_path = './report'
    pickle_folder_path = './report'
    for folder in third_level_folders:
        folder_path = os.path.join(report_folder_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        if folder == 'picture':
            picture_folder_path = folder_path  
        else:
            pickle_folder_path = folder_path
    
    data_clean_rlt, df_cleaned = data_clean_overview.run(autocap_data_raw, pulse_data_raw, picture_folder_path, pickle_folder_path)
    # alg_rlt_list.append(data_clean_rlt)
    
    if data_clean_rlt['ErrorCode'][0] == 0 and len(df_cleaned) > 0:
        # 容量计算
        soh_info = cal_soh_dsoc_dcap.get_results(data_clean_rlt['out']['容量计算需求数据'][0])

        print("开始计算plus")
        first_order.analyse_cell_plus_data(data_clean_rlt['out']['脉冲数据'], 1, 5, 150)
        print("结束计算plus")

        # full_cap = float(data_clean_rlt['out']['额定容量'][0])
        # # 计算电池使用年限
        # if data_clean_rlt['out']['生产日期'][0] != '/' and data_clean_rlt['out']['生产日期'][0] != 'N/A':
        #     year_cal = int(datetime.now().strftime("%Y")) - int(data_clean_rlt['out']['生产日期'][0])
        # else:
        #     year_cal = 0
        # so_file_path = r"./lib/electro_chemistry_cuda.so"  
        
        # this_csv_path = './'
        # soh_rlt = SOH_electro.run(df_cleaned=df_cleaned,
        #                                      full_cap=full_cap,
        #                                      cell_type=data_clean_rlt['out']['电池类型'][0],
        #                                      so_path = so_file_path,
        #                                      result_save_path= this_csv_path,
        #                                      cells_for_ods=1, year= year_cal, alg_ah=data_clean_rlt['out']['统计估算容量'][0], rate_total_vol = data_clean_rlt['out']['额定总电压'][0] ,
        #                                      min_volt = data_clean_rlt['out']['单体最低电压'][0], max_volt= data_clean_rlt['out']['单体最高电压-最大值'][0], charge_efficiency = data_clean_rlt['out']['电桩充电效率'][0],
        #                                      soc_start = data_clean_rlt['out']['起始SOC'][0],
        #                                      soc_end = data_clean_rlt['out']['结束SOC'][0])
        

        # # 1. 一致性检查
        # consis_rlt = consistency.run(df_cleaned,data_clean_rlt['out']['电池类型'][0] ,data_clean_rlt['out']['单体最高电压-最大值'][0],\
        #                              data_clean_rlt['out']['单体最低电压'][0],\
        #                              data_clean_rlt['out']['温度极差'][0],\
        #                              data_clean_rlt['out']['温度极差-最低列'][0],\
        #                              picture_folder_path, pickle_folder_path, soh_rlt, data_clean_rlt)
        # alg_rlt_list.append(consis_rlt)
        # # 2. 滥用工况评估
        # abusive_condition_result = abusive_condition.run(df_cleaned,data_clean_rlt['out']['电池类型'][0], full_cap, picture_folder_path, pickle_folder_path,
        #                                                  vol_max= data_clean_rlt['out']['BMS单体最高允许充电电压'][0], temp_max= data_clean_rlt['out']['BMS最高允许温度'][0],
        #                                                  voltage_abnormal_value_ratio= data_clean_rlt['out']['电压值异常比例'][0], 
        #                                                  temp_abnormal_value_ratio = data_clean_rlt['out']['温度值异常比例'][0],
        #                                                  soc_abnormal_value_ratio = data_clean_rlt['out']['SOC异常值比例'][0]
        #                                                 )
        # alg_rlt_list.append(abusive_condition_result)
        # # 3. BMS传感器有效性评估
        # sensor_check_result = sensor_check.run(df_cleaned, data_clean_rlt['out']['电池类型'][0], picture_folder_path, pickle_folder_path)
        # alg_rlt_list.append(sensor_check_result)
        # # 4. 动力电池容量评估结果保存
        # alg_rlt_list.append(soh_rlt)
        # # 5. 充电安全性评估
        # safe_judge_rlt= safe_judge.run(df_cleaned, data_clean_rlt['out']['电池类型'][0], abusive_condition_result['out']['电池组充电工况数据'][0], picture_folder_path, pickle_folder_path, \
        #                                temp_sorce = consis_rlt['out']['电池组温度一致性评分'][0])
        # alg_rlt_list.append(safe_judge_rlt)
        # alg_rlt_list.append(filename[index]) # 记录文件名
        # # 检测结果输出到json文件
        # rlt_json = generate_json.run(df_cleaned, alg_rlt_list, report_folder_path) 
        # return rlt_json
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
