# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from   openpyxl import Workbook
from   openpyxl.utils import get_column_letter
from   openpyxl.styles import Border, Side, PatternFill, Alignment, Font
import os
import json
import traceback
from tool_utils.pvlog import Logger
import datetime
level = "error"
log = Logger('./logs/pile_json.log', level=level)






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


def create_info(key_name, value1, value2, value3, value4, value5, value6="是"):
    """
    创建电池信息字典

    :param alg_rlt: 算法结果列表
    :param item_name: 项目名称，如"电池类型"或"充电总时长"
    :return: 字典结构
    """
    return {
        key_name: {
            "结果值": str(value1),
            "异常提示": value2,
            "异常原因": value3,
            "单位": value4,
            "指标说明": value5,
            "是否展示": value6
        }
    }


def run(df, alg_rlt_list, report_save_path,sn):
    """
            generate excel report :

            :param df_cleaned:  数据清洗算法输出的df数据序列
            :param alg_rlt_list: 所有算法模块输出结果
            :param report_save_path: 输出报告保存路径
            :return:   rlt_res = {
                                "code_id": 200,
                                "describe": "export excel report",
                                "data": {},
                                "level": "",
                                "summary": [],
                                "table": [],
                                "ErrorCode": [0, 0, '']}，

    """
    rlt_res = {
        "code_id": alg_rlt_list[0]['ErrorCode'][0],
        "sn": "",
        "serial_number": "",
        
        "result": "", # 检测结果
        "pass": [],  # 检测通过的项
        "fail": [],  # 检测失败的项
        "data": {},
        
        "level": "",  # 报告等级
        "summary": [],
        "ErrorCode": [0, 0, ''],
        "describe": "export json"
        
    }

    try:
        rlt_res["data"] = {
            # "car_info": {},
            # "battery_info": {},
            # "charging_info": {},
            # "charging_management_system_info":{}     # 充电桩相关统计信息
        }

        rlt_res['summary'] = 'ver1.0_bulid_20250513'
        rlt_res['sn'] = sn[5:21]
        rlt_res['serial_number'] = sn
               
        if alg_rlt_list[0]['ErrorCode'][0] == 0:
            check_list = {'电池类型_BAT_TYPE': True, 
                        '电池组额定容量_RATE_SOH': True, 
                        '电池组额定总压_RATE_VOL': True, 
                        # '电池组循环次数_CYCLE': True, 
                        '车辆识别码_VIN': True, 
                        'BMS允许充电单体电压_BMS_ALW_MAX_VOL': True, 
                        'BMS允许最大充电电流_BMS_ALW_MAX_CUR':     True, 
                        '电池组额定能量_RATE_KWH': True, 
                        'BMS允许最高充电总电压_BMS_ALW_MAX_VOL': True, 
                        'BMS允许最高温度_BMS_ALW_TEMP': True, 
                        '充电桩输出最高电压_CHARGE_MAX_VOL':True, 
                        '充电桩输出最低电压_CHARGE_MIN_VOL':True, 
                        '充电桩输出最高电流_CHARGE_MAX_CUR':True, 
                        '充电桩输出最低电流_CHARGE_MIN_CUR':True, 
                        "soc":True,
                        "BMS总电压_vol_total":True,
                        "充电桩输出电压_charge_out_vol":True,
                        "BMS总电流_current":True,
                        "充电桩输出电流_charge_out_cur":True,
                        "最高单体电池电压_vol_max":True,
                        "最高单体电池节号_max_vol_no":True,

                        "最低单体电池电压_vol_min":True,
                        "最高温度_temp_max":True,
                        "最高温度点探头号_max_temp_no":True,
                        "最低温度_temp_min":True,
                        "最低温度点探头号_min_temp_no":True,
                        "充电桩枪温_charge_gun_temp":True
            }

            same_data_radio = 0.8 # 同一数据判定阈值
            # 数据上传准确性检测
            '''BMS上传电池Pack以及充电握手信息''' 
            
            confidence = '正常'
            explanation = ''
            reason = '' 
            if 'BAT_TYPE' in df.columns:
                
                if int(df['BAT_TYPE'][0]) == 3 or int(df['BAT_TYPE'][0]) == 6:
                    confidence = '正常'
                else:
                    confidence = '异常'
                    explanation = f"电池类型为{df['BAT_TYPE'][0]}，电池类型应该为3或者6"
                    reason = 'BMS未上传或数据解析异常'
            else:
                confidence = '异常'
                explanation = '缺少BAT_TYPE列'
                reason = 'BMS未上传或数据解析异常'
                check_list['电池类型_BAT_TYPE'] = False
            
            rlt_res["data"].update(
                create_info("电池类型_BAT_TYPE",confidence ,explanation,
                            reason, '', ''))
            
            
            confidence = '正常'
            explanation = ''
            reason = '' 
            
            charge_ah_by_soc = 0
            if alg_rlt_list[0]["out"]["额定容量"][0] == '/' or int(alg_rlt_list[0]["out"]["额定容量"][0]) < 1:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['额定容量'][0]}，电池组额定容量应该大于0"
                reason = 'BMS未上传或数据解析异常' 
                check_list['电池组额定容量_RATE_SOH'] = False
            else:
                charge_ah_by_soc = int(alg_rlt_list[0]["out"]["额定容量"][0])*(int(alg_rlt_list[0]['out']['结束SOC'][0]) - int(alg_rlt_list[0]['out']['起始SOC'][0]))/100

            rlt_res["data"].update(
                create_info("电池组额定容量_RATE_SOH", confidence, explanation,
                            reason, '安时(Ah)', ''))

            
            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['额定总电压'][0] == '/' or int(alg_rlt_list[0]['out']['额定总电压'][0]) < 50:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['额定总电压'][0]}，电池组额定总压介于100~1000V之间"
                reason = 'BMS未上传或数据解析异常'
                check_list['电池组额定总压_RATE_VOL'] = False

            rlt_res["data"].update(
                create_info("电池组额定总压_RATE_VOL", confidence, explanation,
                            reason, '伏(V)', ''))
            
            
            # confidence = '正常'
            # explanation = ''
            # reason = ''
            # if alg_rlt_list[0]['out']['电池充电次数'][0] == '/' or int(alg_rlt_list[0]['out']['电池充电次数'][0]) < 1 or int(alg_rlt_list[0]['out']['电池充电次数'][0]) > 65530:
            #     confidence = '异常'
            #     explanation = f"读取值为{alg_rlt_list[0]['out']['电池充电次数'][0]}，电池组循环次数介于1-3000之间"
            #     reason = 'BMS未上传或数据解析异常'
            #     check_list['电池组循环次数_CYCLE'] = False

            # rlt_res["data"].update(
            #     create_info("电池组循环次数_CYCLE", confidence, explanation,
            #                 reason, '', ''))

            confidence = '正常'
            explanation = ''
            reason = ''
            if len(str(alg_rlt_list[0]['out']['车辆识别码'][0])) != 17:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['车辆识别码'][0]}，车辆VIN码上传不正确"
                reason = 'BMS未上传或数据解析异常'
                check_list['车辆识别码_VIN'] = False

            rlt_res["data"].update(
                create_info("车辆识别码_VIN", confidence, explanation,
                            reason, '', ''))


            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['BMS单体最高允许充电电压'][0] == '/' or int(alg_rlt_list[0]['out']['BMS单体最高允许充电电压'][0]) < 1 or int(alg_rlt_list[0]['out']['BMS单体最高允许充电电压'][0]) > 5:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['BMS单体最高允许充电电压'][0]}，单体最高电压范围介于3.5-4.5之间"
                reason = 'BMS未上传或数据解析异常'
                check_list['BMS允许充电单体电压_BMS_ALW_MAX_VOL'] = False
    
            rlt_res["data"].update(
                create_info("BMS允许充电单体电压_BMS_ALW_MAX_VOL", confidence, explanation,
                            reason, '伏(V)', ''))
            

            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['BMS允许最高充电电流'][0] == '/' or int(alg_rlt_list[0]['out']['BMS允许最高充电电流'][0]) < 1 or int(alg_rlt_list[0]['out']['BMS允许最高充电电流'][0]) > 1000:
                confidence = '异常'
                value =  4* int(alg_rlt_list[0]["out"]["额定容量"][0])
                explanation = f"读取值为{alg_rlt_list[0]['out']['BMS允许最高充电电流'][0]}，BMS允许最大充电电流介于50~{value}A之间"
                reason = 'BMS未上传或数据解析异常'
                check_list['BMS允许最大充电电流_BMS_ALW_MAX_CUR'] = False
            

            rlt_res["data"].update(
                create_info("BMS允许最大充电电流（BMS_ALW_MAX_CUR）", confidence, explanation,
                            reason, '安(A)', ''))
            
            confidence = '正常'
            explanation = ''
            reason = ''
            value =  int(alg_rlt_list[0]["out"]["额定容量"][0])*int(alg_rlt_list[0]['out']['额定总电压'][0])/1000
            if alg_rlt_list[0]['out']['额定总能量'][0] == '/' or abs(float(alg_rlt_list[0]['out']['额定总能量'][0]) - value) > 10:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['额定总能量'][0]}，BMS额定总能量应该大致为{value}千瓦时(KW·h)"
                reason = 'BMS未上传或数据解析异常'
                check_list['电池组额定能量_RATE_KWH'] = False

            rlt_res["data"].update(
                create_info("电池组额定能量_RATE_KWH", confidence, explanation,
                            reason, '千瓦时(KW·h)', ''))
            
            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['BMS最高允许充电总电压'][0] == '/' or int(alg_rlt_list[0]['out']['BMS最高允许充电总电压'][0]) < int(alg_rlt_list[0]['out']['额定总电压'][0]) or int(alg_rlt_list[0]['out']['BMS最高允许充电总电压'][0]) > int(alg_rlt_list[0]['out']['额定总电压'][0])*1.5:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['BMS最高允许充电总电压'][0]}，BMS最高允许充电总电压应介于{alg_rlt_list[0]['out']['额定总电压'][0]}~{int(alg_rlt_list[0]['out']['额定总电压'][0])*1.5}V之间"
                reason = 'BMS未上传或数据解析异常'
                check_list['BMS允许最高充电总电压_BMS_ALW_MAX_VOL'] = False

            rlt_res["data"].update(
                create_info("BMS允许最高总电压_BMS_ALW_TVOL", confidence, explanation,
                            reason, '伏(V)', ''))
            
            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['BMS最高允许温度'][0] == '/' or  int(alg_rlt_list[0]['out']['BMS最高允许温度'][0]) < 50 or  int(alg_rlt_list[0]['out']['BMS最高允许温度'][0]) >85:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['BMS最高允许温度'][0]}，BMS最高允许温度应该介于 50 ~ 85℃之间"
                reason = 'BMS未上传或数据解析异常'
                check_list['BMS允许最高温度_BMS_ALW_TEMP'] = False

            rlt_res["data"].update(
                create_info("BMS允许最高温度_BMS_ALW_TEMP", confidence, explanation,
                            reason, '摄氏度(℃)',''))
            
            
            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['电桩输出最高电压'][0] == '/' or int(alg_rlt_list[0]['out']['电桩输出最高电压'][0]) < int(alg_rlt_list[0]['out']['电桩输出最低电压'][0]) or int(alg_rlt_list[0]['out']['电桩输出最高电压'][0]) > 1000:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['电桩输出最高电压'][0]}，充电桩输出最高电压应高于充电桩输出最低电压，且一般不高于1000V"
                reason = '充电桩未上传或数据解析异常' 
                check_list['充电桩输出最高电压_CHARGE_MAX_VOL'] = False

            rlt_res["data"].update(
                create_info("充电桩输出最高电压_CHARGE_MAX_VOL", confidence, explanation,
                            reason, '伏(V)', ''))
            
            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['电桩输出最低电压'][0] == '/' or int(alg_rlt_list[0]['out']['电桩输出最低电压'][0]) > int(alg_rlt_list[0]['out']['电桩输出最高电压'][0]) or int(alg_rlt_list[0]['out']['电桩输出最高电压'][0]) < 50:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['电桩输出最低电压'][0]}，充电桩输出最低电压应小于充电桩输出最高电压，且大于50V"
                reason = '充电桩未上传或数据解析异常'
                check_list['充电桩输出最低电压_CHARGE_MIN_VOL'] = False

            rlt_res["data"].update(
                create_info("充电桩输出最低电压_CHARGE_MIN_VOL", confidence, explanation,
                            reason, '伏(V)', ''))
            
            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['电桩输出最大电流'][0] == '/' or int(alg_rlt_list[0]['out']['电桩输出最大电流'][0]) < int(alg_rlt_list[0]['out']['电桩输出最大电流'][0]) or int(alg_rlt_list[0]['out']['电桩输出最大电流'][0]) > 1000:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['电桩输出最大电流'][0]}，充电桩输出最大电流应高于充电桩输出最小电流，且一般不高于1000A"
                reason = '充电桩未上传或数据解析异常' 
                check_list['充电桩输出最高电流_CHARGE_MAX_CUR'] = False

            rlt_res["data"].update(
                create_info("充电桩输出最高电流_CHARGE_MAX_CUR", confidence, explanation,
                            reason, '安(A)', ''))
            
            confidence = '正常'
            explanation = ''
            reason = ''
            if alg_rlt_list[0]['out']['电桩输出最小电流'][0] == '/' or int(alg_rlt_list[0]['out']['电桩输出最小电流'][0]) > int(alg_rlt_list[0]['out']['电桩输出最大电流'][0]) or int(alg_rlt_list[0]['out']['电桩输出最小电流'][0]) <1:
                confidence = '异常'
                explanation = f"读取值为{alg_rlt_list[0]['out']['电桩输出最小电流'][0]}，充电桩输出最小电流应小于充电桩输出最大电流，且大于1A"
                reason = '充电桩未上传或数据解析异常'
                check_list['充电桩输出最低电流_CHARGE_MIN_CUR'] = False
    
            rlt_res["data"].update(
                create_info("充电桩输出最低电流_CHARGE_MIN_CUR", confidence, explanation,
                            reason, '安(A)', ''))
            

            '''充电统计信息'''
            confidence = '正常'
            explanation = ''
            reason = ''
            
            diff_soc = df['soc'].diff().dropna()
            indices = diff_soc.loc[(diff_soc > 5) | (diff_soc < -5)].index
            if len(indices) > 0 and indices[0] > 1:
                confidence = '异常'
                explanation = '电池soc序列，出现异常跳变，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'
                check_list['soc'] = False

            rlt_res["data"].update(
                create_info("soc", confidence, explanation,
                            reason, '(%)', ''))
            
            confidence = '正常'
            explanation = ''
            reason = ''
            mode_count = df['vol_total'].value_counts().max()
            mode_ratio = mode_count / len( df['vol_total'])
            
            if mode_ratio >= same_data_radio:
                confidence = '异常'
                explanation = f'电池总电压vol_total，相同值占比超过{same_data_radio*100}%，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'
                check_list['BMS总电压_vol_total'] = False

            rlt_res["data"].update(
                create_info("BMS总电压_vol_total", confidence, explanation,
                            reason, '伏(V)', ''))
        
            confidence = '正常'
            explanation = ''
            reason = ''
            # 1. 在BMS电压上传变化的基础上，检查充电桩电压上传是否正常
            if mode_ratio < same_data_radio:
                diff_vol = abs(df['vol_total'] - df['charge_out_vol']).mean()
                if diff_vol > 50:
                    confidence = '异常'
                    explanation = f'BMS电压与充电桩电压检测压差平均值超过50V，数据上传不正确'
                    reason = '充电桩电压上传不正确'
                    check_list['充电桩电压_charge_out_vol'] = False

                elif diff_vol > 10:
                    # 检查充电桩输出电压与BMS输出电压的相关性，若相关性小于0.5，则说明BMS输出电压有问题，否则，两者输出均可能存在偏差
                    corr_coef = df['vol_total'].corr(df['charge_out_vol'])
                    if corr_coef < 0.5:
                        confidence = '异常'
                        explanation = f'BMS电压与充电桩电压相关性小于0.5，BMS或充电桩电压上传不正确'
                        reason = 'BMS上传错误或数据解析异常'
                        check_list['充电桩电压_charge_out_vol'] = False
            
            rlt_res["data"].update(
                create_info("充电桩输出电压_charge_out_vol", confidence, explanation,
                            reason, '伏(V)', ''))
            
            # 检测BMS输出电流是否高于BMS需求电流 
            confidence = '正常'
            explanation = ''
            reason = ''
            diff_cur = df['current'] - df['bms_need_cur']
            mode_count = (diff_cur >0).sum()   
            mode_ratio = mode_count / len( diff_cur)
            if mode_ratio > same_data_radio:
                confidence = '异常'
                explanation = f'电池总电流current大于BMS需求电流bms_need_cur，占比超过{same_data_radio*100}%，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'
                check_list['BMS总电流_current'] = False

            rlt_res["data"].update(
                create_info("BMS总电流_current", confidence, explanation,
                            reason, '安(A)', ''))
            
            # 检测BMS输出电流与充电桩输出电流相关性
            confidence = '正常'
            explanation = ''
            reason = ''
            diff_cur = abs(df['current'] - df['charge_out_cur']).mean()
            if diff_cur > 20:
                confidence = '异常'
                explanation = f'BMS检测电流与充电桩输出电流差平均值超过50A，数据上传不正确'
                if charge_ah_by_soc != 0:
                    d1 =  abs(int(alg_rlt_list[0]['out']['累计充电容量'][0])- charge_ah_by_soc)
                    d2 =  abs(int(alg_rlt_list[0]['out']['电桩累计充电容量'][0])- charge_ah_by_soc)
                    if d1 > d2:
                        reason = 'BMS上传错误或数据解析异常'
                        check_list['BMS总电流_current'] = False
                        rlt_res["data"].update(
                            create_info("BMS总电流_current", confidence, explanation,
                                        reason, '安(A)', ''))
                    else:
                        reason = '充电桩上传错误或数据解析异常'
                        check_list['充电桩输出电流_charge_out_cur'] = False
                        rlt_res["data"].update(
                                create_info("充电桩输出电流_charge_out_cur", confidence, explanation,
                                            reason, '安(A)', ''))

                else:
                    # 检查充电桩输出电压与BMS输出电压的相关性，若相关性小于0.5，则说明BMS输出电压有问题，否则，两者输出均可能存在偏差
                    corr_coef = df['current'].corr(df['charge_out_cur'])
                    if corr_coef < 0.5:
                        confidence = '异常'
                        explanation = f'BMS电流与充电桩输出电流相关性小于0.5，BMS或充电桩电流上传不正确'
                        reason = 'BMS上传错误或数据解析异常'
                        check_list['充电桩电流_charge_out_cur'] = False
            
                    rlt_res["data"].update(
                        create_info("充电桩输出电流_charge_out_cur", confidence, explanation,
                                    reason, '安(A)', ''))

            confidence = '正常'
            explanation = ''
            reason = ''
            mode_count = df['vol_max'].value_counts().max()
            mode_ratio = mode_count / len( df['vol_max'])
            if mode_ratio > same_data_radio:
                confidence = '异常'
                explanation = f'最高单体电池电压序列值vol_max， 相同值占比超过{same_data_radio*100}%，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'
                check_list['最高单体电池电压_vol_max'] = False

            rlt_res["data"].update(
                create_info("最高单体电池电压_vol_max", confidence, explanation,
                            reason, '伏(V)', ''))
            
            
            confidence = '正常'
            explanation = ''
            reason = '' 
            mode_count = (df['max_vol_no'] < 1).sum()
            mode_ratio = mode_count / len( df['max_vol_no'])
            if mode_ratio > same_data_radio:
                confidence = '异常'
                explanation = f'最高单体电池节号max_vol_no， 0值占比超过{same_data_radio*100}%，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'
                check_list['最高单体电池节号_max_vol_no'] = False

            rlt_res["data"].update(
                create_info("最高单体电池节号_max_vol_no", confidence, explanation,
                            reason, '', ''))
            
            
            confidence = '正常'
            explanation = ''
            reason = ''
            min_vol = df['vol_min'].iloc[0]
            if np.isnan(min_vol) or float(min_vol) < 0.1:
                confidence = '异常'
                explanation = f'最低单体电池电压vol_min为{min_vol}， 数据上传不正确, 正常值范围 3.0-4.5V'
                reason = 'BMS上传错误或数据解析异常'
                check_list['最低单体电池电压_vol_min'] = False

            rlt_res["data"].update(
                create_info("最低单体电池电压_vol_min", confidence, explanation,
                            reason, '伏(V)', ''))
            
            confidence = '正常'
            explanation = ''
            reason = '' 
            mode_count = df['temp_max'].value_counts().max()
            mode_ratio = mode_count / len( df['temp_max'])
            if mode_ratio > same_data_radio:
                confidence = '异常'
                explanation = f'最高温度序列值temp_max， 相同值占比超过{same_data_radio*100}%，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'
                check_list['最高温度_temp_max'] = False

            rlt_res["data"].update(
                create_info("最高温度_temp_max", confidence,explanation,
                            reason, '摄氏度(℃)', ''))

            confidence = '正常'
            explanation = ''
            reason = '' 
            mode_count = (df['max_temp_no'] < 1).sum()
            mode_ratio = mode_count / len( df['max_temp_no'])
            if mode_ratio > same_data_radio:
                confidence = '异常'
                explanation = f'最高温度探头号max_temp_no， 0值占比超过{same_data_radio*100}%，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'
                check_list['最高温度点探头号_max_temp_no'] = False 

            rlt_res["data"].update(
                create_info("最高温度点探头号_max_temp_no`", confidence, explanation,
                            reason, '', ''))
            
            
            confidence = '正常'
            explanation = ''
            reason = '' 
            mode_count = df['temp_min'].value_counts().max()
            mode_ratio = mode_count / len( df['temp_min'])
            if mode_ratio > same_data_radio:
                confidence = '异常'
                explanation = f'最低温度序列值temp_min， 相同值占比超过{same_data_radio*100}%，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'
                check_list['最低温度_temp_min'] = False

            rlt_res["data"].update(
                create_info("最低温度_temp_min", confidence , explanation,
                            reason, '摄氏度(℃)', ''))

            confidence = '正常'
            explanation = ''
            reason = '' 
            mode_count = (df['min_temp_no'] < 1).sum()
            mode_ratio = mode_count / len( df['min_temp_no'])
            if mode_ratio > same_data_radio:
                confidence = '异常'
                explanation = f'最低温度探头号min_temp_no， 0值占比超过{same_data_radio*100}%，数据上传不正确'
                reason = 'BMS上传错误或数据解析异常'   
                check_list['最低温度点探头号_min_temp_no'] = False 

            rlt_res["data"].update(
                create_info("最低温度点探头号_min_temp_no", confidence, explanation,
                            reason, '', ''))
            
            confidence = '正常'
            explanation = ''
            reason = '' 
            mode_count = df['charge_gun_temp'].value_counts().max()
            mode_ratio = mode_count / len( df['charge_gun_temp'])
            if mode_ratio > same_data_radio:
                confidence = '异常'
                explanation = f'枪温序列值charge_gun_temp， 相同值占比超过{same_data_radio*100}%，数据上传不正确'
                reason = '充电桩上传错误或数据解析异常'
                check_list['充电桩枪温_charge_gun_temp'] = False

            rlt_res["data"].update(
                create_info("充电桩枪温_charge_gun_temp", confidence, explanation,
                            reason, '安(A)', ''))

            # 充电桩检测计算
            rlt_res["data"].update(
                create_info("电桩输出充电度数", alg_rlt_list[0]['out']['电桩累计充电度数'][0],
                            alg_rlt_list[0]['out']['电桩累计充电度数'][1],
                            [], 'kwh', '',"否"))
            rlt_res["data"].update(
                create_info("充电转换效率", alg_rlt_list[0]['out']['电桩充电效率'][0],
                            alg_rlt_list[0]['out']['电桩充电效率'][1],
                            [], '%', '',"否"))
            rlt_res["data"].update(
                create_info("枪温检测有效性", "/",
                            "缺乏关键数值，不支持分析",
                            [], '', '',"否"))
            # 输出最终检测结果
            if check_list['电池组额定容量_RATE_SOH'] == False or  check_list['soc'] == False or check_list['BMS总电压_vol_total'] == False \
            or check_list['充电桩输出电压_charge_out_vol'] == False or check_list['BMS总电流_current'] == False or check_list['充电桩输出电流_charge_out_cur'] == False \
            or check_list['最高单体电池电压_vol_max'] == False:
                rlt_res['result'] = '不合格'
            else:
                rlt_res['result'] = '合格'
            
            # 输出合格项和非合格项
            rlt_res['pass'] = [k for k, v in check_list.items() if v == True]
            rlt_res['fail'] = [k for k, v in check_list.items() if v == False]
        else:
            rlt_res['result'] = '不合格'
            rlt_res['fail'] = alg_rlt_list[0]['ErrorCode'][2]

        # 将数据写入JSON文件
        file_path = os.path.join(report_save_path, sn +'.json')
        with open(file_path, 'w', encoding='utf-8') as json_file:
             json.dump(rlt_res, json_file, ensure_ascii=False, indent=4)  
        
        rlt_res['ErrorCode'][0] = 0
        rlt_res['ErrorCode'][2] = "生成电桩数据检测报告成功"
        return rlt_res
    except Exception as e:
        log.logger.error(f"json report generate error: {traceback.format_exc()}") 
        rlt_res['ErrorCode'][0] = -99
        rlt_res['ErrorCode'][2] = f"json report generate error: {traceback.format_exc()}"
        return rlt_res
