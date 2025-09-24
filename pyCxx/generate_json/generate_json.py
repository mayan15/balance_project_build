# -*- coding: utf-8 -*-

import pandas as pd
from   openpyxl import Workbook
from   openpyxl.utils import get_column_letter
from   openpyxl.styles import Border, Side, PatternFill, Alignment, Font
import os
import json
import traceback
from tool_utils.pvlog import Logger
import datetime
level = "error"
log = Logger('./logs/json.log', level=level)

NO_DEBUG = False

reference_range = \
    {
        "电池类型": ['磷酸铁锂', '三元锂'],
        "充电时长": ['0.5-8.0'],
        "总电压": ['100-800'],
        "充电电流": ['1-300'],
        "温度": ['<60'],
        "SOC": ['0-100'],
        "单体电压": ['2.8-4.3'],
        "电压极差": ['<0.1'],
        "温度极差": ['<15'],
        "电压上升速率": ['<0.1'],
        "温度上升速率": ['<5'],
        "快充时间比例": ['<50'],
        "异常值比例": ['<3']
    }



def judge_value(value, limit_max, limit_min):
    """
        检查 DataFrame 中是否存在指定的列名, 并判断该列存在的值是否正常

        :param df: 输入的 DataFrame
        :param column_names: 要检查的列名列表
        :limit_max: 最大有效值
        :limit_min: 最小有效值
        :need: 某些值不需要判断是否属于正常范围
    """
    confidence  = ''
    explanation = ''
    
    if value != '/' and value != 'N/A':
        if value > limit_max:
            confidence = '↑'
        elif value < limit_min:
            confidence = '↓'
    else:
        confidence = 'N/A'
        explanation = '数值不支持分析'
    return confidence, explanation


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
            "参考范围": value3,
            "单位": value4,
            "指标说明": value5,
            "是否展示": value6
        }
    }


def run(data_clean_rlt, alg_rlt_list, report_save_path):
    """
            generate excel report :

            :param df_cleaned:  数据清洗算法输出的全部数据
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
        "describe": "export json",
        "data": {},
        "level": "",  # 报告等级
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }

    try:
        rlt_res["data"] = {
            "car_info":{},

            "battery_info":{},

            "pulse_info": {},

            "soh_info": {},

            "balance_info": {},

            # 额外增加检测说明和维保建议
            "检测说明": {},
            "维保建议":{},
        }

        rlt_res['summary'] = 'ver1.0_build_20250912'
        rlt_res['table'] = [
                            {'过高':f"计算值高于指标阈值{1.2}倍或以上"},
                            {'高':f"计算值高于指标阈值{0.8}-{1.2}倍之间"},
                            {'偏高':f"计算值高于指标阈值{0.3}-{0.8}倍之间"},
                            {'正常':'计算值在指标值阈值范围内'},
                            {'偏低':f"计算值低于指标阈值{0.2}-{0.5}倍之间"},
                            {'低':f"计算值低于指标阈值{0.5}-{0.8}倍之间"},
                            {'过低':f"计算值低于指标阈值{0.8}倍或以上"}
                        ]

        '''车辆信息'''
        rlt_res["data"]['car_info'].update(create_info("vin", data_clean_rlt['out']['vin'][0], '', '/', '', ''))
        rlt_res["data"]['car_info'].update(create_info("行驶里程", data_clean_rlt['out']['mileage'][0], '', '/', '公里(km)', ''))
        rlt_res["data"]['car_info'].update(create_info("设备ID", data_clean_rlt['out']['device_id'][0], '', '/', '', '', '否'))
       
        '''电池包基本信息''' 
        cell_type =  ''
        if data_clean_rlt['out']['battery_type'][0] == 'NCM':
            cell_type = '三元锂'
        elif data_clean_rlt['out']['battery_type'][0] == 'LFP':
            cell_type = '磷酸铁锂'
        rlt_res["data"]['battery_info'].update(create_info("额定容量", data_clean_rlt['out']['battery_capacity'][0], '', '/', '安时(Ah)', ''))
        rlt_res["data"]['battery_info'].update(create_info("电池类型", cell_type, '', '三元锂(NCM),磷酸铁锂(LFP)', '', ''))
        # rlt_res["data"]['battery_info'].update(create_info("额定总压", data_clean_rlt['out']['battery_voltage'][0], '', '/', '伏(V)', ''))
        rlt_res["data"]['battery_info'].update(create_info("额定能量", data_clean_rlt['out']['battery_rata_energy'][0], '', '/', '千瓦时(KW·h)', ''))
        rlt_res["data"]['battery_info'].update(create_info("出厂日期", data_clean_rlt['out']['battery_manufacture_date'][0], '', '/', '', ''))
        rlt_res["data"]['battery_info'].update(create_info("电池厂商", data_clean_rlt['out']['battery_manufacture'][0], '', '/', '', ''))
        
        '''均衡结果'''
        rlt_res["data"]['balance_info'].update(create_info("测试电芯号", alg_rlt_list[3]['out']['均衡测试电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡前电压值", alg_rlt_list[3]['out']['均衡前电压值'][0], '', '/', '伏(V)', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡后电压值", alg_rlt_list[3]['out']['均衡后电压值'][0], '', '/', '伏(V)', ''))    
        rlt_res["data"]['balance_info'].update(create_info("均衡充放电容量", alg_rlt_list[3]['out']['均衡容量'][0], '', '/', '安时(Ah)', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡前最大压差", alg_rlt_list[3]['out']['均衡前最大压差'][0], '', '/', '毫伏(mV)', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡后最大压差", alg_rlt_list[3]['out']['均衡后最大压差'][0], '', '/', '毫伏(mV)', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡前电压极值", alg_rlt_list[3]['out']['均衡前电压极值'][0], '', '/', '', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡后电压极值", alg_rlt_list[3]['out']['均衡后电压极值'][0], '', '/', '', ''))
        # rlt_res["data"]['balance_info'].update(create_info("均衡前最高单体电压值", alg_rlt_list[3]['out']['均衡前最高单体电压值'][0], '', '/', '伏(V)', ''))
        # rlt_res["data"]['balance_info'].update(create_info("均衡前最高单体电芯号", alg_rlt_list[3]['out']['均衡前最高单体电芯号'][0], '', '/', '', ''))
        # rlt_res["data"]['balance_info'].update(create_info("均衡前最低单体电压值", alg_rlt_list[3]['out']['均衡前最低单体电压值'][0], '', '/', '伏(V)', ''))
        # rlt_res["data"]['balance_info'].update(create_info("均衡前最低单体电芯号", alg_rlt_list[3]['out']['均衡前最低单体电芯号'][0], '', '/', '', ''))
        # rlt_res["data"]['balance_info'].update(create_info("均衡后最高单体电压值", alg_rlt_list[3]['out']['均衡后最高单体电压值'][0], '', '/', '伏(V)', ''))
        # rlt_res["data"]['balance_info'].update(create_info("均衡后最高单体电芯号", alg_rlt_list[3]['out']['均衡后最高单体电芯号'][0], '', '/', '', ''))
        # rlt_res["data"]['balance_info'].update(create_info("均衡后最低单体电压值", alg_rlt_list[3]['out']['均衡后最低单体电压值'][0], '', '/', '伏(V)', ''))
        # rlt_res["data"]['balance_info'].update(create_info("均衡后最低单体电芯号", alg_rlt_list[3]['out']['均衡后最低单体电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡测试容差占比", alg_rlt_list[3]['out']['均衡测试容差占比'][0], '', '/', '百分比(%)', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡前一致性异常电芯号", alg_rlt_list[3]['out']['均衡前一致性异常电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['balance_info'].update(create_info("均衡测试容差最大值", alg_rlt_list[3]['out']['均衡测试容差最大值'][0], '', '/', '安时(Ah)', '', '否'))
        rlt_res["data"]['balance_info'].update(create_info("均衡测试容差占比最大值", alg_rlt_list[3]['out']['均衡测试容差占比最大值'][0], '', '/', '百分比(%)', '', '否'))
        rlt_res["data"]['balance_info'].update(create_info("一致性计算异常说明", alg_rlt_list[3]['out']['一致性计算异常说明'][0], '', '/', '', '', '否'))
        
        balance_summary = alg_rlt_list[3]['out']['说明'][0]
        balance_advice = alg_rlt_list[3]['out']['建议'][0]

        '''内阻测试结果'''
        rlt_res["data"]['pulse_info'].update(create_info("测试电芯号", alg_rlt_list[1]['out']['内阻计算电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['pulse_info'].update(create_info("内阻测试内阻值", alg_rlt_list[1]['out']['内阻计算直流内阻值'][0],'', '/', '毫欧(mΩ)', ''))
        rlt_res["data"]['pulse_info'].update(create_info("内阻测试计算tau值", alg_rlt_list[1]['out']['内阻计算tau值'][0], '', '/', '秒(s)', '', '是'))
        rlt_res["data"]['pulse_info'].update(create_info("内阻测试计算R0值", alg_rlt_list[1]['out']['内阻计算R0值'][0], '', '/', '毫欧(mΩ)', '', '否'))
        rlt_res["data"]['pulse_info'].update(create_info("内阻测试计算R1值", alg_rlt_list[1]['out']['内阻计算R1值'][0], '', '/', '毫欧(mΩ)', '', '否'))
        rlt_res["data"]['pulse_info'].update(create_info("内阻测试起始电压", alg_rlt_list[1]['out']['内阻计算电芯起始电压值'][0], '', '/', '伏(V)', ''))
        rlt_res["data"]['pulse_info'].update(create_info("内阻测试阻值异常电芯号", alg_rlt_list[1]['out']['内阻值异常电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['pulse_info'].update(create_info("内阻测试计算异常说明", alg_rlt_list[1]['out']['内阻计算异常说明'][0], '', '/', '', '', '否'))
        pulse_summary = alg_rlt_list[1]['out']['说明'][0]
        pulse_advice = alg_rlt_list[1]['out']['建议'][0]
        
        '''SOH结果'''
        # 包括SOH最高值，SOH最低值，SOH最高值电芯号，SOH最低值电芯号，SOH异常电芯号，容量测试电芯号，容量测试SOH值，SOH异常维修建议，SOH结果说明
        rlt_res["data"]['soh_info'].update(create_info("测试电芯号", alg_rlt_list[2]['out']['容量测试电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['soh_info'].update(create_info("容量测试SOH值", alg_rlt_list[2]['out']['容量测试SOH值'][0], '', '/', '百分比(%)', ''))
        rlt_res["data"]['soh_info'].update(create_info("容量测试SOH最高值", alg_rlt_list[2]['out']['SOH最高值'][0], '', '/', '百分比(%)', ''))
        rlt_res["data"]['soh_info'].update(create_info("容量测试SOH最低值", alg_rlt_list[2]['out']['SOH最低值'][0], '', '/', '百分比(%)', ''))
        rlt_res["data"]['soh_info'].update(create_info("容量测试SOH最高值电芯号", alg_rlt_list[2]['out']['SOH最高值电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['soh_info'].update(create_info("容量测试SOH最低值电芯号", alg_rlt_list[2]['out']['SOH最低值电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['soh_info'].update(create_info("容量测试SOH值异常电芯号", alg_rlt_list[2]['out']['SOH值异常电芯号'][0], '', '/', '', ''))
        rlt_res["data"]['soh_info'].update(create_info("容量测试SOH计算异常说明", alg_rlt_list[2]['out']['SOH计算异常说明'][0], '', '/', '', '', '否'))
        soh_summary = alg_rlt_list[2]['out']['说明'][0]
        soh_advice = alg_rlt_list[2]['out']['建议'][0]


        # 说明及建议
        summary = "仅针对本次测试数据的分析结果进行说明，内容仅供参考。"
        # summary += "若某项检测结果为 'N/A'，是因为本次测试数据不支持该项出具分析结果。"

        rlt_res["data"]['检测说明'].update(create_info("总述", summary, '', '', '', ''))
        rlt_res["data"]['检测说明'].update(create_info("均衡测试", balance_summary, '', '', '', ''))
        rlt_res["data"]['检测说明'].update(create_info("内阻测试", pulse_summary, '', '', '', ''))
        rlt_res["data"]['检测说明'].update(create_info("容量测试", soh_summary, '', '', '', ''))
        
        rlt_res["data"]['维保建议'].update(create_info("均衡测试", balance_advice, '', '', '', ''))
        rlt_res["data"]['维保建议'].update(create_info("内阻测试", pulse_advice, '', '', '', ''))
        rlt_res["data"]['维保建议'].update(create_info("容量测试", soh_advice, '', '', '', ''))

        # 将数据写入JSON文件
        file_path = os.path.join(report_save_path, 'data.json')
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(rlt_res, json_file, ensure_ascii=False, indent=4)  

        rlt_res['ErrorCode'][0] = 0
        rlt_res['ErrorCode'][2] = "生成报告成功"
        return rlt_res
    except Exception as e:
        log.logger.error(f"json report generate error: {traceback.format_exc()}") 
        rlt_res['ErrorCode'][0] = -99
        rlt_res['ErrorCode'][2] = f"json report generate error: {traceback.format_exc()}"
        return rlt_res
