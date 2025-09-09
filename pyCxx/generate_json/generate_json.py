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


def run(df_cleaned, alg_rlt_list, report_save_path):
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
        "describe": "export json",
        "data": {},
        "level": "",  # 报告等级
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }

    try:
        rlt_res["data"] = {
            "car_info": {},

            "battery_info": {},

            "charging_info": {},
           
            "soh_info":{},

            "voltage_info": {},

            "current_info": {},

            "temperature_info": {},

            "abusive_info": {},

            "sensor_info": {},

            "alarm_management_info": {},

            "score_info":{},

            # 额外增加检测说明和维保建议
            "检测说明": {},
            "维保建议":{},

            "charging_management_system_info":{}     # 充电桩相关统计信息

        }

        rlt_res['summary'] = 'ver1.0_bulid_20250813'
        rlt_res['table'] = [
                            {'过高':f"计算值高于指标阈值{1.2}倍或以上"},
                            {'高':f"计算值高于指标阈值{0.8}-{1.2}倍之间"},
                            {'偏高':f"计算值高于指标阈值{0.3}-{0.8}倍之间"},
                            {'正常':'计算值在指标值阈值范围内'},
                            {'偏低':f"计算值低于指标阈值{0.2}-{0.5}倍之间"},
                            {'低':f"计算值低于指标阈值{0.5}-{0.8}倍之间"},
                            {'过低':f"计算值低于指标阈值{0.8}倍或以上"}
                        ]
       
        '''电车基本信息'''
        rlt_res["data"]['car_info'].update(
            create_info("车辆识别码", alg_rlt_list[0]['out']['车辆识别码'][0], alg_rlt_list[0]['out']['车辆识别码'][1],
                        '/', '', ''))
        rlt_res["data"]['car_info'].update(
            create_info("生产厂商", alg_rlt_list[0]['out']['生产厂家'][0], alg_rlt_list[0]['out']['生产厂家'][1],
                        '/', '', ''))
        
        rlt_res["data"]['car_info'].update(
            create_info("车辆制造商", alg_rlt_list[0]['out']['车辆制造商'][0], alg_rlt_list[0]['out']['车辆制造商'][1],
                        '/', '', '')) 
        rlt_res["data"]['car_info'].update(
            create_info("生产日期", alg_rlt_list[0]['out']['生产日期'][0], alg_rlt_list[0]['out']['生产日期'][1],
                        '/', '', ''))
        '''电池包基本信息''' 
        cell_type =  ''
        if alg_rlt_list[0]['out']['电池类型'][0] == 'NCM':
            cell_type = '三元锂'
        elif alg_rlt_list[0]['out']['电池类型'][0] == 'LFP':
            cell_type = '磷酸铁锂'
        rlt_res["data"]['battery_info'].update(
            create_info("电池类型", cell_type, alg_rlt_list[0]['out']['电池类型'][1],
                        '三元锂(NCM),磷酸铁锂(LFP)', '', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("电芯数量", alg_rlt_list[0]['out']['电芯数量'][0], alg_rlt_list[0]['out']['电芯数量'][1],
                        '/', '节', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("额定容量", alg_rlt_list[0]['out']['额定容量'][0], alg_rlt_list[0]['out']['额定容量'][1],
                        '/', '安时(Ah)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("额定能量", alg_rlt_list[0]['out']['额定总能量'][0], alg_rlt_list[0]['out']['额定总能量'][1],
                        '/', '千瓦时(KW·h)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("循环充电次数", alg_rlt_list[0]['out']['电池充电次数'][0], alg_rlt_list[0]['out']['电池充电次数'][1],
                        '0-5000', '次', ''))
        
        # 2025/2/4 new add 
        rlt_res["data"]['battery_info'].update(
            create_info("额定总压", alg_rlt_list[0]['out']['额定总电压'][0], alg_rlt_list[0]['out']['额定总电压'][1],
                        '/', '伏(V)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("BMS允许最高单体电压", alg_rlt_list[0]['out']['BMS单体最高允许充电电压'][0], alg_rlt_list[0]['out']['BMS单体最高允许充电电压'][1],
                        '/', '伏(V)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("BMS允许最高电流", alg_rlt_list[0]['out']['BMS允许最高充电电流'][0], alg_rlt_list[0]['out']['BMS允许最高充电电流'][1],
                        '/', '安(A)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("BMS允许最高总电压", alg_rlt_list[0]['out']['BMS最高允许充电总电压'][0], alg_rlt_list[0]['out']['BMS最高允许充电总电压'][1],
                        '/', '伏(V)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("BMS允许最高温度", alg_rlt_list[0]['out']['BMS最高允许温度'][0], alg_rlt_list[0]['out']['BMS最高允许温度'][1],
                        '/', '摄氏度(℃)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("充电桩输出最高电压", alg_rlt_list[0]['out']['电桩输出最高电压'][0], alg_rlt_list[0]['out']['电桩输出最高电压'][1],
                        '/', '伏(V)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("充电桩输出最低电压", alg_rlt_list[0]['out']['电桩输出最低电压'][0], alg_rlt_list[0]['out']['电桩输出最低电压'][1],
                        '/', '伏(V)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("充电桩输出最高电流", alg_rlt_list[0]['out']['电桩输出最大电流'][0], alg_rlt_list[0]['out']['电桩输出最大电流'][1],
                        '/', '安(A)', ''))
        rlt_res["data"]['battery_info'].update(
            create_info("充电桩输出最低电流", alg_rlt_list[0]['out']['电桩输出最小电流'][0], alg_rlt_list[0]['out']['电桩输出最小电流'][1],
                        '/', '安(A)', ''))
        
        '''充电统计信息'''
        rlt_res["data"]['charging_info'].update(
            create_info("充电SOC区间", f"{alg_rlt_list[0]['out']['起始SOC'][0]}-{alg_rlt_list[0]['out']['结束SOC'][0]}",
                        '',
                        '0-100', '%', ''))
        rlt_res["data"]['charging_info'].update(
            create_info("充电总电压区间", f"{alg_rlt_list[0]['out']['起始总电压'][0]}-{alg_rlt_list[0]['out']['结束总电压'][0]}",
                        '',
                        '100-1000', '伏(V)', ''))
        
        rlt_res["data"]['charging_info'].update(
            create_info("最大单体电压区间", f"{alg_rlt_list[0]['out']['单体最高电压-最小值'][0]}-{alg_rlt_list[0]['out']['单体最高电压-最大值'][0]}",
                        '',
                        '2.0-4.5', '伏(V)', ''))
        
        rlt_res["data"]['charging_info'].update(
            create_info("充电电流区间", f"{alg_rlt_list[0]['out']['最小充电电流'][0]}-{alg_rlt_list[0]['out']['最大充电电流'][0]}",
                        '',
                        f'0-{alg_rlt_list[0]["out"]["BMS允许最高充电电流"][0]}', '安(A)', ''))
        
        rlt_res["data"]['charging_info'].update(
            create_info("充电结束电流", alg_rlt_list[0]['out']['结束充电电流'][0],
                        '',
                        f'0-{alg_rlt_list[0]["out"]["BMS允许最高充电电流"][0]}', '安(A)', '', "否"))
        
        rlt_res["data"]['charging_info'].update(
            create_info("温度区间", f"{alg_rlt_list[0]['out']['温度最小值'][0]}-{alg_rlt_list[0]['out']['温度最大值'][0]}",
                        '',
                        f'0-{alg_rlt_list[0]["out"]["BMS最高允许温度"][0]}', '摄氏度(℃)', ''))
        
        
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['累计充电容量'][0], alg_rlt_list[0]["out"]["额定容量"][0], 1.0)
        rlt_res["data"]['charging_info'].update(
            create_info("累计充电容量", alg_rlt_list[0]['out']['累计充电容量'][0],                    
                        confidence,
                        f'1.0-{alg_rlt_list[0]["out"]["额定容量"][0]}', '安时(Ah)', ''))
        
        rlt_res["data"]['charging_info'].update(
            create_info("累计充电时长", alg_rlt_list[0]["out"]["充电总时长"][0],                    
                        confidence,
                        f'10.0-{300}', '分钟(Min)', ''))
      
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['累计充电度数'][0], alg_rlt_list[0]["out"]["额定总能量"][0], 1.0)
        rlt_res["data"]['charging_info'].update(
            create_info("累计充电度数", alg_rlt_list[0]['out']['累计充电度数'][0],
                        confidence,
                        f'1.0-{alg_rlt_list[0]["out"]["额定总能量"][0]}', '千瓦时(KW·h)', ''))
        
        if alg_rlt_list[4]["out"]["可行驶里程估算"][0][0] == 'N/A':
            rlt_res["data"]['charging_info'].update(
                create_info("预估可行驶里程数", f'{alg_rlt_list[0]["out"]["电桩行驶里程"][0][0]}-{alg_rlt_list[0]["out"]["电桩行驶里程"][0][1]}',
                            '',
                            '/', '公里(KM)', ''))
        else:
            rlt_res["data"]['charging_info'].update(
                create_info("预估可行驶里程数", f'{alg_rlt_list[4]["out"]["可行驶里程估算"][0][0]}-{alg_rlt_list[4]["out"]["可行驶里程估算"][0][1]}',
                            '',
                            '/', '公里(KM)', ''))
            
        
        rlt_res["data"]['charging_info'].update(
            create_info("充电工况特征值列表", alg_rlt_list[0]["out"]["充电工况特征值"][0],
                        '',
                        '', '', '', "否"))

        # 2025/2/24 new add
        rlt_res["data"]['charging_info'].update(
            create_info("最高温度变化分布描述", alg_rlt_list[0]["out"]["最高温度变化特征值"][0],
                        '',
                        '', '', '',"否"))

        rlt_res["data"]['charging_info'].update(
            create_info("充电线阻值", alg_rlt_list[0]["out"]["line_res"][0],
                        alg_rlt_list[0]["out"]["line_res"][1],
                        alg_rlt_list[0]["out"]["line_res"][2], '', '',"否"))

        # 2025/2/6 new add 
        rlt_res["data"]['charging_info'].update(
            create_info("充电桩输出电压与bms检查电压差值统计描述", alg_rlt_list[0]["out"]["total_vol_dec_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电桩输出电流与bms检查电流差值统计描述", alg_rlt_list[0]["out"]["total_cur_dec_describe"][0],
                        '',
                        '', '', '',"否"))
        
        rlt_res["data"]['charging_info'].update(
            create_info("充电单体最高与平均值差统计描述", alg_rlt_list[0]["out"]["vol_dec_mid_and_max_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电最高电压值统计描述", alg_rlt_list[0]["out"]["vol_total_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电总电流值统计描述", alg_rlt_list[0]["out"]["current_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电单体最高电压值统计描述", alg_rlt_list[0]["out"]["vol_max_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电最高温度值统计描述", alg_rlt_list[0]["out"]["temp_max_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电最低温度值统计描述", alg_rlt_list[0]["out"]["temp_min_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电桩输出电压值统计描述", alg_rlt_list[0]["out"]["charge_out_vol_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电桩输出电流值统计描述", alg_rlt_list[0]["out"]["charge_out_cur_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电单体节号分布描述", alg_rlt_list[0]["out"]["max_vol_no_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电最高温度探头号分布描述", alg_rlt_list[0]["out"]["max_temp_no_describe"][0],
                        '',
                        '', '', '',"否"))
        rlt_res["data"]['charging_info'].update(
            create_info("充电最低温度探头号分布描述", alg_rlt_list[0]["out"]["min_temp_no_describe"][0],
                        '',
                        '', '', '',"否"))
    
        
        '''电池性能检测'''
        if alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0] != 'N/A':
            
            confidence, explanation = judge_value(alg_rlt_list[4]['out']['动力电池组剩余容量'][0], alg_rlt_list[0]["out"]["额定容量"][0], round(0.75*alg_rlt_list[0]["out"]["额定容量"][0],2))
            # rlt_res["data"]['soh_info'].update(
            #     create_info("蓄电池最高单体电芯剩余容量", alg_rlt_list[4]['out']['动力电池组剩余容量'][0],
            #                 confidence,
            #                 f'{round(0.6*alg_rlt_list[0]["out"]["额定容量"][0],2)}-{alg_rlt_list[0]["out"]["额定容量"][0]}', '安时(Ah)', ''))
            
            rlt_res["data"]['soh_info'].update(
                create_info("电池组剩余容量", alg_rlt_list[4]['out']['动力电池组剩余容量'][0],
                            confidence,
                            f'{round(0.75*alg_rlt_list[0]["out"]["额定容量"][0],2)}-{alg_rlt_list[0]["out"]["额定容量"][0]}', '安时(Ah)', ""))

            confidence, explanation = judge_value(alg_rlt_list[4]['out']['动力电池组剩余总能量'][0], alg_rlt_list[0]["out"]["额定总能量"][0], round(0.75*alg_rlt_list[0]["out"]["额定总能量"][0],2))
            
            rlt_res["data"]['soh_info'].update(
                create_info("电池组剩余总能量", alg_rlt_list[4]['out']['动力电池组剩余总能量'][0],
                            confidence,
                            f'{round(0.75*alg_rlt_list[0]["out"]["额定总能量"][0],2)}-{alg_rlt_list[0]["out"]["额定总能量"][0]}', '千瓦时(KW·h)', ""))
            
            confidence, explanation = judge_value(alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0], 100, 75)
            # rlt_res["data"]['soh_info'].update(
            #     create_info("蓄电池平均容量保有率", alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0],
            #                 confidence,
            #                 '60-100', '%', ''))

            rlt_res["data"]['soh_info'].update(
                create_info("剩余容量百分比（SOH）", alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0],
                            confidence,
                            '75-100', '%', ''))
           
            confidence, explanation = judge_value(alg_rlt_list[4]['out']['动力电池能量损失速率'][0], 5, 0)
            rlt_res["data"]['soh_info'].update(
                create_info("能量损失速率", alg_rlt_list[4]['out']['动力电池能量损失速率'][0],
                            confidence,
                            '0-5.0', '度/年(KW·h/year)', ""))
            
        # 未计算出相关SOH参数信息
        else:
            rlt_res["data"]['soh_info'].update(
                create_info("电池组剩余容量", 'N/A',
                            '数据不足，无法计算',
                            f'{round(0.75*alg_rlt_list[0]["out"]["额定容量"][0],2)}-{alg_rlt_list[0]["out"]["额定容量"][0]}', '安时(Ah)', ""))
            
            # rlt_res["data"]['soh_info'].update(
            #     create_info("最高单体电芯剩余容量", 'N/A',
            #                 '数据不足，无法计算',
            #                 f'{round(0.6*alg_rlt_list[0]["out"]["额定容量"][0],2)}-{alg_rlt_list[0]["out"]["额定容量"][0]}', '安时(Ah)', ''))

            rlt_res["data"]['soh_info'].update(
                create_info("电池组剩余总能量", 'N/A',
                            '数据不足，无法计算',
                            f'{round(0.75*alg_rlt_list[0]["out"]["额定总能量"][0],2)}-{alg_rlt_list[0]["out"]["额定总能量"][0]}', '千瓦时(KW·h)', ""))
            
           
            rlt_res["data"]['soh_info'].update(
                create_info("剩余容量百分比（SOH）", 'N/A',
                            '数据不足，无法计算',
                            '75-100', '%', ""))

            rlt_res["data"]['soh_info'].update(
                create_info("容量损失速率", 'N/A',
                            '数据不足，无法计算',
                            '0-5.0', '度/年(KW·h/year)', ""))
            
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['总电压上升速率'][0], 2, 0.01)
        rlt_res["data"]['voltage_info'].update(
            create_info("总压最大上升速率", alg_rlt_list[0]['out']['总电压上升速率'][0],
                        confidence,
                        '0.01-2.0', '伏/秒(V/s)', ""))
        
        if alg_rlt_list[0]['out']['电池类型'][0] == "LFP":  # 磷酸
            lmin = 0.35
        else:
            lmin = 0.2
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['单体电压极差'][0], lmin, 0.01)
        rlt_res["data"]['voltage_info'].update(
            create_info("单体电压最大差值", alg_rlt_list[0]['out']['单体电压极差'][0],
                        confidence,
                        f'0.01-{lmin}', '伏(V)', ""))
        
       
        
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['最高单体电压上升速率'][0], 15, 1.0)
        rlt_res["data"]['voltage_info'].update(
            create_info("单体电压最大上升速率", alg_rlt_list[0]['out']['最高单体电压上升速率'][0],
                        confidence,
                        '1.0-15.0', '毫伏/秒(mV/s)', ""))
        
        confidence, explanation = judge_value(alg_rlt_list[1]['out']['电池组电压一致性评分'][0], 100, 60)

        if alg_rlt_list[1]['out']['电池组电压一致性评分'][0] !='N/A':
            rlt_res["data"]['voltage_info'].update(
                create_info("电压一致性", round(alg_rlt_list[1]['out']['电池组电压一致性评分'][0]/100,2),
                            confidence,
                            '0.6-1.0', '', ""))
        else:
            rlt_res["data"]['voltage_info'].update(
                create_info("电压一致性", 'N/A',
                            '',
                            '0.6-1.0', '', ""))
            
        # 新增SOC一致性评分
        confidence, explanation = judge_value(alg_rlt_list[1]['out']['电池组SOC一致性评分'][0], 100, 60)
        if alg_rlt_list[1]['out']['电池组SOC一致性评分'][0] !='N/A':
            rlt_res["data"]['voltage_info'].update(
                create_info("SOC一致性", round(alg_rlt_list[1]['out']['电池组SOC一致性评分'][0]/100,2),
                            confidence,
                            '0.6-1.0', '', ""))
        else:
            rlt_res["data"]['voltage_info'].update(
                create_info("SOC一致性", 'N/A',
                            '',
                            '0.6-1.0', '', ""))
            
        
        confidence, explanation = judge_value(alg_rlt_list[1]['out']['电池组容压比一致性评分'][0], 100, 60)
        rlt_res["data"]['voltage_info'].update(
            create_info("容压比一致性", round(alg_rlt_list[1]['out']['电池组容压比一致性评分'][0]/100,2),
                        confidence,
                        '0.6-1.0', '', ""))

        

        
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['电流上升速率'][0], 15, 0.1)
        rlt_res["data"]['current_info'].update(
            create_info("最大电流上升速率", alg_rlt_list[0]['out']['电流上升速率'][0],
                        confidence,
                        '0.1-15.0', '安/秒(A/s)', ''))
        
        if alg_rlt_list[2]['out']['电池组充电工况数据'][0][5] != 'N/A':
            high_current_time = round(alg_rlt_list[2]['out']['电池组充电工况数据'][0][5]*alg_rlt_list[0]['out']['充电总时长'][0]/100,2)
        else:
            high_current_time = 'N/A'
        confidence, explanation = judge_value(high_current_time, 0.9*alg_rlt_list[0]['out']['充电总时长'][0], 0)
        rlt_res["data"]['current_info'].update(
            create_info("快充时长", high_current_time,
                        confidence,
                        f'{0}-{round(0.9*alg_rlt_list[0]["out"]["充电总时长"][0])}', '分钟(Min)', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['温度极差'][0], 15, 0)
        rlt_res["data"]['temperature_info'].update(
            create_info("温度探头最大差值", alg_rlt_list[0]['out']['温度极差'][0],
                        confidence,
                        '1-15', '摄氏度(℃)', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['最高温度上升速率'][0], 0.2, 0.01)
        rlt_res["data"]['temperature_info'].update(
            create_info("温度探头最大上升速率", alg_rlt_list[0]['out']['最高温度上升速率'][0],
                        confidence,
                        '0.01-0.2', '摄氏度/秒(℃/s)', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[1]['out']['电池组温度一致性评分'][0], 100, 60)

        if alg_rlt_list[1]['out']['电池组温度一致性评分'][0] != 'N/A':
            rlt_res["data"]['temperature_info'].update(
                create_info("温度一致性", round(alg_rlt_list[1]['out']['电池组温度一致性评分'][0]/100,2),
                            confidence,
                            '0.6-1.0', '', ""))
        else:
            rlt_res["data"]['temperature_info'].update(
                create_info("温度一致性", 'N/A',
                            confidence,
                            '0.6-1.0', '', ""))
        
        
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['电压值异常比例'][0], 10, 0)
        rlt_res["data"]['abusive_info'].update(
            create_info("电压值异常比例", alg_rlt_list[0]['out']['电压值异常比例'][0],
                        confidence,
                        '0-10', '%', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[2]['out']['电池组充电工况数据'][0][4], 10, 0)
        rlt_res["data"]['abusive_info'].update(
            create_info("电流异常值比例", alg_rlt_list[2]['out']['电池组充电工况数据'][0][4],
                        confidence,
                        '0-10', '%', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['温度值异常比例'][0], 10, 0)
        rlt_res["data"]['abusive_info'].update(
            create_info("温度值异常比例", alg_rlt_list[0]['out']['温度值异常比例'][0],
                        confidence,
                        '0-10', '%', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[0]['out']['SOC异常值比例'][0], 10, 0)
        rlt_res["data"]['abusive_info'].update(
            create_info("SoC值异常比例", alg_rlt_list[0]['out']['SOC异常值比例'][0],
                        confidence,
                        '0-10', '%', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[5]['out']['输出电流异常值占比'][0], 10, 0)
        rlt_res["data"]['abusive_info'].update(
            create_info("充电桩输出电流异常比例", alg_rlt_list[5]['out']['输出电流异常值占比'][0],
                        confidence,
                        '0-10', '%', ''))
        
        '''车载BMS系统检测'''
        confidence, explanation = judge_value(alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][0], 100, 50)

        if alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][0] != 'N/A':
            rlt_res["data"]['sensor_info'].update(
                create_info("电压测量值准确性", round(alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][0]/100,2),
                            confidence,
                            '0.5-1.0', '', ''))
        else:
            rlt_res["data"]['sensor_info'].update(
                create_info("电压测量值准确性", 'N/A',
                            confidence,
                            '0.5-1.0', '', ''))
        
        # confidence, explanation = judge_value(alg_rlt_list[3]['out']['BMS温度传感器有效性评分'][0], 100, 20)

        # if alg_rlt_list[3]['out']['BMS温度传感器有效性评分'][0] != 'N/A':
        #     rlt_res["data"]['sensor_info'].update(
        #         create_info("温度传感器有效性", round(alg_rlt_list[3]['out']['BMS温度传感器有效性评分'][0]/100,2),
        #                     confidence,
        #                     '0.2-1.0', '', ''))
        # else:
        #     rlt_res["data"]['sensor_info'].update(
        #         create_info("温度传感器有效性", 'N/A',
        #                     confidence,
        #                     '0.2-1.0', '', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][0], 100, 50)
        
        if alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][0] != 'N/A':
            rlt_res["data"]['sensor_info'].update(
                create_info("电流测量值准确性", round(alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][0]/100,2),
                            confidence,
                            '0.5-1.0', '', ''))
        else:
            rlt_res["data"]['sensor_info'].update(
                create_info("电流测量值准确性", 'N/A',
                            confidence,
                            '0.5-1.0', '', ''))
        
        # BMS相关状态上传
        rlt_res["data"]['alarm_management_info'].update(
            create_info("单体电池电压报警", alg_rlt_list[0]['out']['单体电池电压报警'][0],
                        "",
                        "正常/过高/过低/未上传", '', ''))
        
        # rlt_res["data"]['alarm_management_info'].update(
        #     create_info("单体电池SOC状态", alg_rlt_list[0]['out']['单体电池SOC状态'][0],
        #                 "",
        #                 "正常/过高/过低/未上传", '', ''))
        
        rlt_res["data"]['alarm_management_info'].update(
            create_info("充电过温", alg_rlt_list[0]['out']['充电过温'][0],
                        "",
                        "正常/过高/过低/未上传", '', ''))
        
        rlt_res["data"]['alarm_management_info'].update(
            create_info("充电过电流", alg_rlt_list[0]['out']['充电过电流'][0],
                        "",
                        "正常/过流/不可信/未上传", '', ''))
        # rlt_res["data"]['alarm_management_info'].update(
        #     create_info("SOC跳变报警", "/",
        #                 "",
        #                 "正常无告警/需告警但未触发/状态错误", '', ''))
        
        rlt_res["data"]['alarm_management_info'].update(
            create_info("电池绝缘状态", alg_rlt_list[0]['out']['电池绝缘状态'][0],
                        "",
                         "正常/不正常/不可信/未上传", '', ''))
        
        rlt_res["data"]['alarm_management_info'].update(
            create_info("输出连接器连接状态", alg_rlt_list[0]['out']['输出连接器连接状态'][0],
                        "",
                        "正常/不正常/不可信/未上传", '', ''))
        
        # rlt_res["data"]['alarm_management_info'].update(
        #     create_info("充电允许", alg_rlt_list[0]['out']['充电允许'][0],
        #                 "",
        #                 "禁止/允许/未上传", '', ''))
        
        rlt_res["data"]['alarm_management_info'].update(
            create_info("BMS中止充电原因", alg_rlt_list[0]['out']['BMS中止充电原因'][0],
                        "",
                        "", '', ''))
        
        rlt_res["data"]['alarm_management_info'].update(
            create_info("充电机中止充电原因", alg_rlt_list[0]['out']['充电机中止充电原因'][0],
                        "",
                        "", '', ''))
        
        '''评分信息'''
 
        confidence, explanation = judge_value(alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0], 100, 60)
        rlt_res["data"]['score_info'].update(
            create_info("电池容量保有率评分", alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0],
                        confidence,
                        '60-100', '', ''))
        # rlt_res["data"]['score_info'].update(
        #     create_info("电池组容量SOH评分", alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0],
        #                 confidence,
        #                 '60-100', '', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[2]['out']['电池组充电工况评分'][0], 100, 60)
        rlt_res["data"]['score_info'].update(
            create_info("充电工况评分", alg_rlt_list[2]['out']['电池组充电工况评分'][0],
                        confidence,
                        '60-100', '', ''))
        

        confidence, explanation = judge_value(alg_rlt_list[1]['out']['电池组一致性总评分'][0], 100, 60) 
        rlt_res["data"]['score_info'].update(
            create_info("电池组性能一致性评分", alg_rlt_list[1]['out']['电池组一致性总评分'][0],
                        confidence,
                        '60-100', '', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[3]['out']['BMS传感器有效性总评分'][0], 100, 60)
        rlt_res["data"]['score_info'].update(
            create_info("车载BMS管理系统评分", alg_rlt_list[3]['out']['BMS传感器有效性总评分'][0],
                        confidence,
                        '60-100', '', ''))
        
        confidence, explanation = judge_value(alg_rlt_list[5]['out']['电池组充电安全评分'][0], 100, 60)
        rlt_res["data"]['score_info'].update(
            create_info("充电安全评分", alg_rlt_list[5]['out']['电池组充电安全评分'][0],
                        confidence,
                        '60-100', '', ''))
        # 计算总评分及对应的评分分级
        # 充电工况和充电安全不作为维护判断标准
        div_ = 1.0
        unable_msg = ""
        if alg_rlt_list[1]['out']['电池组一致性总评分'][0] == "N/A":
            voltage_score = 0
            div_ -= 0.3
            unable_msg += "【电池组一致性】"
        else:
            voltage_score = alg_rlt_list[1]['out']['电池组一致性总评分'][0]

        if alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0] == "N/A": 
            soh_score = 0
            div_ -= 0.3
            unable_msg += "【容量SOH】"
        else:
            soh_score = alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0]

        if alg_rlt_list[3]['out']['BMS传感器有效性总评分'][0] == "N/A":
            sensor_score = 0
            div_ -= 0.2
            unable_msg += "【BMS传感器有效性】"
        else:   
            sensor_score = alg_rlt_list[3]['out']['BMS传感器有效性总评分'][0]

        if alg_rlt_list[5]['out']['电池组充电安全评分'][0] == "N/A":
            safe_score = 0
            div_ -= 0.1
        else:
            safe_score = alg_rlt_list[5]['out']['电池组充电安全评分'][0]

        if alg_rlt_list[2]['out']['电池组充电工况评分'][0] == "N/A":
            abusive_score = 0
            div_ -= 0.1
            unable_msg += "【充电工况】"
        else:
            abusive_score = alg_rlt_list[2]['out']['电池组充电工况评分'][0]

        # if div_ == 1.0 or (div_ > 0.6 and alg_rlt_list[4]['out']['动力电池组容量保有率评分'][0] != "N/A"):
        if div_ == 1.0 or div_ > 0.6:
            total_score = (voltage_score*0.3 + abusive_score*0.1 + safe_score*0.1 \
                    + sensor_score*0.2 + soh_score*0.3)/div_
        else:
            total_score = 'N/A'

        # else:
        #     if div_ > 0.6:
        #         total_score = (voltage_score*0.3 + abusive_score*0.1 + safe_score*0.1 \
        #             + sensor_score*0.2 + soh_score*0.3)/div_
        #     else:
        #         total_score = 'N/A'

        if  total_score == 'N/A': 
            advice = '无效'
        else:
            if total_score >= 90:
                advice = '优良'
            elif total_score >= 80:
                advice = '良好'
            elif total_score >= 60:
                advice = '中等'
            else:   
                advice = '中下'
        # 总评语
        # 检测结果描述
        rlt = {'检测说明': '', '维保建议': ''}

        # 电池充电工况检测说明
        abusive_summary = "【充电情况】："
        if len(alg_rlt_list[2]['out']['电池组充电工况评分'][1]) > 0:
            abusive_summary += "本次充电时长为 " + str(alg_rlt_list[0]['out']['充电总时长'][0]) + "分钟，充电电量为 " + str(alg_rlt_list[0]['out']['累计充电度数'][0]) + "度" 
            for item in alg_rlt_list[2]['out']['电池组充电工况评分'][1]:
                abusive_summary += "，" + item
            abusive_summary += "。" + "\n"
        else:
            abusive_summary += "[-]" + "\n"


        rlt_res["data"]['检测说明'].update(
            create_info("充电情况", abusive_summary[7:-1],
                        '',
                        '', '', ''))

        # 电池组一致性检测说明    
        consis_summary = "【电池组一致性】：" 
        if len(alg_rlt_list[1]['out']['电池组电压一致性评分'][1]) == 0 and  len(alg_rlt_list[1]['out']['电池组温度一致性评分'][1]) == 0 and len(alg_rlt_list[1]['out']['电池组SOC一致性评分'][1]) == 0:
            consis_summary += "[-]" + "\n"
        else:
            judge = 0
            if len(alg_rlt_list[1]['out']['电池组电压一致性评分'][1]) > 0:
                consis_summary += alg_rlt_list[1]['out']['电池组电压一致性评分'][1][0] 
                judge = 1
            
            if len(alg_rlt_list[1]['out']['电池组温度一致性评分'][1]) == 1:   
                if judge == 1:
                    consis_summary +=  "；"+ alg_rlt_list[1]['out']['电池组温度一致性评分'][1][0] 
                else:
                    consis_summary += alg_rlt_list[1]['out']['电池组温度一致性评分'][1][0]
                judge = 2
            elif len(alg_rlt_list[1]['out']['电池组温度一致性评分'][1]) == 2:
                if judge == 1:
                    consis_summary +=  "；"+ alg_rlt_list[1]['out']['电池组温度一致性评分'][1][0] + "；" +  alg_rlt_list[1]['out']['电池组温度一致性评分'][1][1]
                else:
                    consis_summary += alg_rlt_list[1]['out']['电池组温度一致性评分'][1][0]+ "；" +  alg_rlt_list[1]['out']['电池组温度一致性评分'][1][1]
                judge = 2
                        
            # if len(alg_rlt_list[1]['out']['电池组电压一致性评分'][2]) == 0 and len(alg_rlt_list[1]['out']['电池组温度一致性评分'][2]) ==0:
            #     consis_summary += "，各项指标正常"
            
            if len(alg_rlt_list[1]['out']['电池组SOC一致性评分'][1]) > 0:
                if judge == 0:
                    consis_summary += alg_rlt_list[1]['out']['电池组SOC一致性评分'][1][0]
                else:
                    consis_summary +=  "；"+ alg_rlt_list[1]['out']['电池组SOC一致性评分'][1][0]

            consis_summary += "。" + "\n"
        
        rlt_res["data"]['检测说明'].update(
            create_info("电池组一致性", consis_summary[9:-1],
                        '',
                        '', '', ''))

        # #  电压一致性检测说明     
        # consis_vol_summary =""
        # for item in alg_rlt_list[1]['out']['电池组电压一致性评分'][1]:
        #     consis_vol_summary+= item+" " 
        # # 温度一致性工况检测说明
        # consis_temp_summary=""
        # for item in alg_rlt_list[1]['out']['电池组温度一致性评分'][1]:
        #     consis_temp_summary+= item+" " 
        # # SOC一致性工况检测说明
        # consis_soc_summary=""
        # for item in alg_rlt_list[1]['out']['电池组SOC一致性评分'][1]:
        #     consis_soc_summary += item+" " 

        # # 容压比一致性工况检测说明
        # consis_cap_summary=""
        # for item in alg_rlt_list[1]['out']['电池组容压比一致性评分'][1]:
        #     consis_cap_summary += item+" " 
        
        
        # 电池组容量
        soh_summary = "【剩余容量百分比（SOH）】："
        if alg_rlt_list[4]['out']['动力电池组容量保有率评分'][1] == '':
           soh_summary += "[-]" + "\n"
        else:
           soh_summary += alg_rlt_list[4]['out']['动力电池组容量保有率评分'][1]+ "。" + "\n"

            # if alg_rlt_list[4]['out']['动力电池组容量保有率评分'][2] == '':
            #     #soh_summary += "。" + "\n"
            # #else:
            #     soh_summary += "，处于健康状态" + "。" + "\n"

        rlt_res["data"]['检测说明'].update(
            create_info("电池组剩余容量百分比（SOH）", soh_summary[15:-1],
                        '',
                        '', '', ''))
        
        # 车载管理系统（BMS）
        bms_summary = "【车载管理系统（BMS）】："
        if len(alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][1]) == 0 and len(alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][1]) == 0 and len(alg_rlt_list[5]['out']['电池组充电安全评分'][1]) == 0:
            bms_summary += "[-]" + "\n"
        else:
            judge = 0
            if len(alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][1]) >0:
                bms_summary += alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][1][0]
                judge = 1
            if len(alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][1]) >0:
                if judge == 0:
                    bms_summary += alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][1][0]
                    judge = 1
                else:
                    bms_summary += "；" + alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][1][0]
                    judge = 2
            
            if len(alg_rlt_list[5]['out']['电池组充电安全评分'][1]) >0:
                if judge == 0:
                    bms_summary += alg_rlt_list[5]['out']['电池组充电安全评分'][1][0]
                else:
                    bms_summary += "；" + alg_rlt_list[5]['out']['电池组充电安全评分'][1][0]
            
            bms_summary += "。" + "\n"

        
        rlt_res["data"]['检测说明'].update(
            create_info("车载管理系统（BMS）", bms_summary[14:-1],
                        '',
                        '', '', ''))

        # # 电压传感器检测说明
        # sensor_vol_summary =""
        # for item in alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][1]:
        #     sensor_vol_summary+= item+" " 
        # sensor_cur_summary =""
        # for item in alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][1]:
        #     sensor_cur_summary+= item+" " 
        # # 温度传感器检测说明
        # # sensor_temp_summary =""
        # # for item in alg_rlt_list[3]['out']['BMS温度传感器有效性评分'][1]:
        # #     sensor_temp_summary += item+" " 
        
        # # 充电工况安全性检测说明
        # safe_summary = ""
        # for item in alg_rlt_list[5]['out']['电池组充电安全评分'][1]:
        #     safe_summary += item+" " 

        
        
        # rlt['检测说明'] = "对该车辆在 " + alg_rlt_list[0]['out']['起始时间'][0] + " 到 " + alg_rlt_list[0]['out']['结束时间'][0] + " 的电池数据分析得出" \
        #               + "充电总时长为 " + str(alg_rlt_list[0]['out']['充电总时长'][0]) + "分钟，总充电电量为 " + str(alg_rlt_list[0]['out']['累计充电度数'][0]) + "度" + "。"+"\n" \
        #               + abusive_summary + "\n"   \
        #               + consis_vol_summary    \
        #               + consis_temp_summary + "\n"  \
        #               + sensor_vol_summary \
        #               + sensor_cur_summary + "\n"\
        #               + consis_soc_summary + "\n"  \
        #               + safe_summary + "\n"  \
        #               + soh_summary 
         
        rlt['检测说明'] = "* 注意：仅针对本次充电数据的分析结果进行说明，内容仅供参考。"  \
                         + "若某项检测内容为【-】，是因为充电数据不支持该项出具分析结果。" + "\n"  \
                         + abusive_summary \
                         + consis_summary \
                         + soh_summary \
                         + bms_summary

        rlt_res["data"]['检测说明'].update(
            create_info("声明", "* 注意：仅针对本次充电数据的分析结果项进行说明，内容仅供参考。"  \
                         + "若某个检测项内容为【-】，是因为充电数据不支持该项出具分析结果。",
                        '',
                        '', '', ''))
        
                                                            
        level_count  = [0, '', '', '', '', '', '', '', '','', '']
        # 维保建议描述
        # 电池充电工况检测说明
        abusive_advice = ""
        if len(alg_rlt_list[2]['out']['电池组充电工况评分'][2]) >0:
            #for item in alg_rlt_list[2]['out']['电池组充电工况评分'][2]:
            abusive_advice = alg_rlt_list[2]['out']['电池组充电工况评分'][2][0] 
            level_count[0] += 1
            level_count[1] = abusive_advice
              
        # 电压一致性工况检测说明
        consis_vol_advice =""
        if len(alg_rlt_list[1]['out']['电池组电压一致性评分'][2]) > 0:
            #for item in alg_rlt_list[1]['out']['电池组电压一致性评分'][2]:
            consis_vol_advice = " " + alg_rlt_list[1]['out']['电池组电压一致性评分'][2][0] 
            level_count[0] += 1
            level_count[2] = consis_vol_advice
                
        # 温度一致性工况检测说明
        consis_temp_advice=""
        if len(alg_rlt_list[1]['out']['电池组温度一致性评分'][2]) > 0:
            for item in alg_rlt_list[1]['out']['电池组温度一致性评分'][2]:
                consis_temp_advice = " " + item 
            level_count[0] += 1
            level_count[3] = consis_temp_advice

        # SOC一致性工况检测说明
        consis_soc_advice=""
        if len(alg_rlt_list[1]['out']['电池组SOC一致性评分'][2]) > 0:
            #for item in alg_rlt_list[1]['out']['电池组SOC一致性评分'][2]:
            consis_soc_advice = " " + alg_rlt_list[1]['out']['电池组SOC一致性评分'][2][0]
            level_count[0] += 1
            level_count[4] = consis_soc_advice

        # # 容压比一致性工况检测说明
        # consis_cap_advice=""
        # if len(alg_rlt_list[1]['out']['电池组容压比一致性评分'][2]) > 0:
        #     for item in alg_rlt_list[1]['out']['电池组容压比一致性评分'][2]:
        #         consis_cap_advice += item+ "\n"
            
        # 电压传感器检测说明
        sensor_vol_advice =""
        if len(alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][2]) > 0:
            #for item in alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][2]:
            sensor_vol_advice = " " + alg_rlt_list[3]['out']['BMS电压传感器有效性评分'][2][0] 
            level_count[0] += 1
            level_count[5] = sensor_vol_advice
        
        # 电流传感器检测说明
        sensor_cur_advice =""
        if len(alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][2]) > 0:
            #for item in alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][2]:
            sensor_cur_advice = " " + alg_rlt_list[3]['out']['BMS电流传感器有效性评分'][2][0]   
            level_count[0] += 1
            level_count[6] = sensor_cur_advice
            

        # 温度传感器检测说明
        # sensor_temp_advice =""
        # if len(alg_rlt_list[3]['out']['BMS温度传感器有效性评分'][2]) > 0:
        #     for item in alg_rlt_list[3]['out']['BMS温度传感器有效性评分'][2]:
        #         sensor_temp_advice +=item+ "\n"
        #     level_count[0] += 1
        #     level_count[7] = sensor_temp_advice

        # 充电安全说明
        safe_advice = ""
        if len(alg_rlt_list[5]['out']['电池组充电安全评分'][2]) > 0:
            #for item in alg_rlt_list[5]['out']['电池组充电安全评分'][2]:
            safe_advice = " " + alg_rlt_list[5]['out']['电池组充电安全评分'][2][0] 
            level_count[0] += 1
            level_count[9] = safe_advice

        # 容量检测说明
        soh_advice = ""
        if len(alg_rlt_list[4]['out']['动力电池组容量保有率评分'][2]) > 0:
            soh_advice = " " + alg_rlt_list[4]['out']['动力电池组容量保有率评分'][2] 
            level_count[8] = soh_advice
            level_count[0] += 1

        
        
        
        rlt_res["level"] = level_count    

        if abusive_advice == "" and consis_vol_advice == "" and consis_temp_advice == "" and consis_soc_advice == "" and sensor_vol_advice == "" and sensor_cur_advice=="" and safe_advice == "" and soh_advice == "":
            # if div_ <1.0:
            #    rlt['维保建议'] = "基于本次充电数据, 您的爱车电池动力系统中可检测部分的功能一切正常。" + "建议每隔1-3个月检测一次，随时关注爱车电池组健康情况。"
            # else:
            rlt['维保建议'] = "本次已检测项中未发现异常项，整体状态良好。建议每隔 1~3 个月进行一次电池充电健康检查，持续关注爱车电池状态，确保驾驶安全和续航性能。"
        else:
            rlt['维保建议'] = "本次检测存在以下异常项:"  \
                        + abusive_advice   \
                        + consis_vol_advice    \
                        + consis_temp_advice \
                        + sensor_vol_advice \
                        + sensor_cur_advice \
                        + consis_soc_advice \
                        + safe_advice \
                        + soh_advice \
                        + "，若您的爱车在历史或后续充电报告中，上述异常项持续累计出现3次及以上，应引起重视，及时联系制造商或4S店对异常项做进一步排查和确认。"
            
        rlt_res["data"]['维保建议'].update(
            create_info("建议", rlt['维保建议'],
                        '',
                        '', '', ''))
        
        
        if total_score == 'N/A':  
            # rlt_res["data"]['score_info'].update(
            #     create_info("总评分", total_score,
            #                 advice,
            #                 [], '', rlt))
            # rlt['维保建议'] = "基于本次充电数据不满足相关检测需求， 本次检测不做任何建议！" 
            rlt_res['ErrorCode'][0] = -9
            rlt_res['ErrorCode'][2] = f"充电数据质量差，{unable_msg}未能出具结果，无法生成报告！"
            return rlt_res     
        else:
            rlt_res["data"]['score_info'].update(
                create_info("总评分", round(total_score, 2),
                            advice,
                            [], '', rlt))
            
        # 充电桩检测
        rlt_res["data"]['charging_management_system_info'].update(
            create_info("电桩输出充电度数", alg_rlt_list[0]['out']['电桩累计充电度数'][0],
                        alg_rlt_list[0]['out']['电桩累计充电度数'][1],
                        [], 'kwh', '',"否"))
        rlt_res["data"]['charging_management_system_info'].update(
            create_info("充电转换效率", alg_rlt_list[0]['out']['电桩充电效率'][0],
                        alg_rlt_list[0]['out']['电桩充电效率'][1],
                        [], '%', '',"否"))
        rlt_res["data"]['charging_management_system_info'].update(
            create_info("枪温检测有效性", "/",
                        "缺乏关键数值，不支持分析",
                        [], '', '',"否"))
        rlt_res["data"]['charging_management_system_info'].update(
            create_info("枪温检测有效性", "/",
                        "缺乏关键数值，不支持分析",
                        [], '', '',"否"))
        rlt_res["data"]['charging_management_system_info'].update(
            create_info("充电枪连接异常", "连接正常",
                        "",
                        [], '', '',"否"))
        rlt_res["data"]['charging_management_system_info'].update(
            create_info("充电保护动作", "/",
                        "缺乏关键数值，不支持分析",
                        [], '', '',"否"))
        rlt_res["data"]['charging_management_system_info'].update(
            create_info("充电桩SN号", alg_rlt_list[6],
                        "",
                        [], '', '',"否"))
       
        #if NO_DEBUG:
        # 将数据写入JSON文件
        file_path = os.path.join(report_save_path, 'data.json')
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(rlt_res, json_file, ensure_ascii=False, indent=4)  
        #else:
            #sp: 新增写入到data文件中，用于测试
        # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join('/root/data/sp/charging_project_build/data/json/', f'{alg_rlt_list[6]}.json')
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
