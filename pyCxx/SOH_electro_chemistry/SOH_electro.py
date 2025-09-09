
import numpy as np  
import pandas as pd

import time
from datetime import datetime  
import pickle

import os
import traceback

from tool_utils.pvlog import Logger

# 创建日志对象
level = "error"
log = Logger('./logs/soh.log', level= level)


"""
    引入主运行模块
"""
from .main_module import run_soh


def make_cols_for_test(df_real, all_temp_names):
    
    if 'time [s]' in df_real.columns:
        pass
    
    elif 'ts_diff' in df_real.columns:
        df_real['date_time'] = df_real['ts_diff']
    else:
        df_real['date_time'] = df_real['time [s]'].copy() * 1000  
        time_counts = []
        for i in range(1, len(df_real)):
            time_counts.append((df_real['date_time'].iloc[i] - df_real['date_time'].iloc[0]) / 1000)
    
        time_counts.insert(0,0)
            
        df_real['time [s]'] = time_counts
        
    print(df_real['time [s]'])
    return df_real

 
"""
    输入数据预处理
"""
def process_index(df):
    
    # df = df.reset_index(drop=False)     # False 为了保留 'date_time'列
    df.index = pd.to_datetime(df["date_time"])    # 不要这行代码是为了保留 'date_time'列
    
    df = df.set_index('date_time', drop=True)
    df['date_time'] = df.index
    
    ts = []
    for i, index in enumerate(df.index):
        # ts.append(pd.to_datetime(df.index[i]).value * 1000)       # 本地测试时用
        # ts.append(pd.to_datetime(df.index[i]).value / 1e6)       # PSS联调读数时要用这个
        ts.append(pd.to_datetime(df['date_time'].iloc[i]).value / 1e6)       # PSS联调读数时要用这个
    
    df['ts'] = ts
    df['ts'] = df['ts'].astype(np.int64)
    df = df.set_index('ts', drop=True)
    
    time_counts = []
    for i in range(1, len(df)):
        time_counts.append((df.index[i] - df.index[0]) / 1000)
    
    time_counts.insert(0,0)
    df['current'] = - df['current']
    df['time [s]'] = time_counts
    # df['time [s]'] = df['time [s]'].astype(np.int64)
    return df
    
def datetime_to_secondsInterval(datetime_in):
    intervals = []
    for i in range(len(datetime_in)):
        delta_time = datetime_in[i] - datetime_in[0]
        total_secs = delta_time.total_seconds()
        intervals.append(total_secs)
    return np.array(intervals)

def check_continuity(start, end, socs, stat):
    check_bars = []

    bar_soc_arr = socs[start:end]
    diff_arr = np.diff(bar_soc_arr)
    seperation_positions = np.where(np.diff(diff_arr) >= 10)[0]
    if len(seperation_positions) == 0:
        check_bars.append([start, end, stat])
    else:
        seperation_positions += 1 + start
        seperation_positions = seperation_positions.tolist() + [end]
        for each_seperation_ind in range(len(seperation_positions)):

            if each_seperation_ind == 0:
                check_bars.append([start, seperation_positions[each_seperation_ind], stat])
            else:
                check_bars.append(
                    [seperation_positions[each_seperation_ind - 1], seperation_positions[each_seperation_ind], stat])

    return check_bars
    
def find_status(cleaned_concat, full_cap):
    # look for status change
    status_change_index = [0]
    for i in range(cleaned_concat.shape[0] - 1):
        if cleaned_concat['status'][i + 1] != cleaned_concat['status'][i]:
            status_change_index.append(i + 1)
    if len(status_change_index) == 1:
        status_change_index.append(cleaned_concat.shape[0] - 1)
    status_change_df = cleaned_concat.iloc[status_change_index]
    status_change_sec = datetime_to_secondsInterval(status_change_df.index)

    # filter out the temperal current change
    status_change_index_ignore = []
    #status_min_keep_time = int(10 * 6)  # 6 timestamps or 1 minutes
    status_min_keep_time = int(10 * 1)  # 10 seconds
    for i in range(len(status_change_sec) - 1):
        if (status_change_sec[i + 1] - status_change_sec[i]) >= status_min_keep_time:
            status_change_index_ignore.append(status_change_index[i])
        if i == len(status_change_sec) - 2:
            status_change_index_ignore.append(status_change_index[i + 1])
    ignore_tiny_change_df = cleaned_concat.iloc[status_change_index_ignore]

    # make status continuous
    status_change_index_merge = [0]
    for i in range(len(ignore_tiny_change_df) - 1):
        if ignore_tiny_change_df['status'][i + 1] == ignore_tiny_change_df['status'][i]:
            pass
        else:
            status_change_index_merge.append(status_change_index_ignore[i + 1])
    if len(status_change_index_merge) == 1:
        status_arr = cleaned_concat["status"].values
        counts = np.bincount(status_arr)
        status_rep = np.argmax(counts)
        status_change_index_merge[0] = np.where(status_arr == status_rep)[0][0]

    merged_change_df = cleaned_concat.iloc[status_change_index_merge]

    # data_clu['簇状态'] has logic error in GEM database!
    # Define new 簇状态 for later usage.
    new_clu_status = []
    for i in range(merged_change_df.shape[0]):
        if merged_change_df['current'][i] < 0.0:
            new_clu_status.append('charge')
        elif merged_change_df['current'][i] > 0.0:
            new_clu_status.append('discharge')
        else:
            new_clu_status.append('hold')

    status_start_index = status_change_index_merge
    status_end_index = status_change_index_merge[1:]
    status_end_index.append(cleaned_concat.shape[0] - 1)
    status_bar = []
    socs = cleaned_concat['soc'].values

    for i in range(len(status_start_index)):
        status_bar += check_continuity(status_start_index[i], status_end_index[i], socs, new_clu_status[i])

    useful_status_bar = []
    for i in range(len(status_bar)):
        if abs(cleaned_concat['current'][status_bar[i][0]:status_bar[i][1]]).mean() > 0.05 * full_cap:
            useful_status_bar.append(status_bar[i])
        else:
            if status_bar[i][2] == 'hold':
                useful_status_bar.append(status_bar[i])
    return useful_status_bar


def separate_status(status_bar, cleaned_concat):
    discharge_bar = []
    charge_bar = []
    hold_bar = []
    for i in range(len(status_bar)):
        try:
            if (status_bar[i][2] == 'discharge'):
                if np.abs(cleaned_concat['soc'][status_bar[i][1] - 1] - cleaned_concat['soc'][status_bar[i][0]]) >= 10:
                    discharge_bar.append(status_bar[i][0:2])
            elif (status_bar[i][2] == 'charge'):
                if np.abs(cleaned_concat['soc'][status_bar[i][1] - 1] - cleaned_concat['soc'][status_bar[i][0]]) >= 10:
                    charge_bar.append(status_bar[i][0:2])
            else:
                if np.abs(cleaned_concat['soc'][status_bar[i][1] - 1] - cleaned_concat['soc'][status_bar[i][0]]) <= 1:
                    hold_bar.append(status_bar[i][0:2])
        except:
            pass
    return discharge_bar, charge_bar, hold_bar
def get_status(x):
    #print(x)
    if x['current'] < 0.0:
        return 2
    elif x['current'] > 0.0:
        return 3
    else:
        return 4


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


def run(df_cleaned, full_cap, cell_type, so_path, result_save_path, cells_for_ods, **kwargs):
    
    rlt_res = {
        "code_id": 99,
        "describe": "soh calculation",
        "out": {},
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }

    df_cleaned['date_time'] = pd.to_datetime(df_cleaned['date_time'])
    df_cleaned.index = pd.to_datetime(df_cleaned["date_time"])
    ''' 采样频率 '''
    time_ = pd.to_datetime(df_cleaned['date_time'])
    # 计算时间差
    time_diff = time_.diff()[1:]  # 去掉第一个 NaT
    # 确保 time_diff 中没有 NaN 或者无效的时间差
    time_diff = time_diff.dropna()
    # 提取秒数，注意: `.seconds` 获取的是时间差的秒数
    deltaT = [td.total_seconds() for td in time_diff]
    # 找出最常见的时间差（即采样频率）
    frequency = pd.Series(deltaT).value_counts().index[0]
    
    ''' 电池类型 '''
    voltage = df_cleaned.filter(like='vol_max')
    voltage.index = df_cleaned['date_time']

    temprature = df_cleaned.filter(like='temp_m')
    temprature.index = df_cleaned['date']

    df_cleaned["status"] = df_cleaned.apply(get_status, axis=1)
    ''' 筛选充放电区间 '''
    status_bar = find_status(df_cleaned, full_cap)
    discharge_bar, charge_ranges, hold_bar = separate_status(status_bar, df_cleaned)
    
    # if voltage.max().max() > 3.7:
    #     battery_type = 'NCM'  # 镍钴锰三元锂电池
    # else:
    #     battery_type = 'LFP'  # 磷酸铁锂电池    
    battery_type = cell_type

    all_volt_names = list(voltage.columns)
    all_temp_names = list(temprature.columns)

    try:
        st = time.time() 
        data = df_cleaned.copy()
        data.index = pd.to_datetime(df_cleaned["date_time"])    # 设置数据的 index 为 date_time
        charge_ranges = [[1,10], [20,25], [30, 35], [40, 45], [50, 55], [56,60], [61,70], [71,80], [0,df_cleaned.shape[0]]] 
        if len(charge_ranges) == 0:
           pass 
        else:     
            '''
                工况筛选二 ： 最后几个充电段选SOC范围最大的
            '''
            # 1-22 18:38更新 - 在倒数后5个充放电段中（不确定这几段数据的时间跨度）选择SOC跨度最大的一段有效充放电数据
            charge_ranges_recent_time = charge_ranges[-5:]
            charge_ranges_select_index = np.array([i[-1]-i[0] for i in charge_ranges_recent_time]).argmax()
            select_range = charge_ranges_recent_time[charge_ranges_select_index]     
            data_piece = data.iloc[select_range[0]: select_range[1], :]        
            data_piece = process_index(data_piece) 
            
            df_temp = data_piece[all_temp_names]
            df_temp['temp (℃)'] = df_temp['temp_max']
            data_piece['temp (℃)'] = df_temp['temp (℃)']
            dec_soc = data['soc'].iloc[-1] - data['soc'][1]
            advice = ''
            summary = ''

            cur_total_error = abs(data['current']- data['charge_out_cur']).median()
            # 2025.2.15 增加对电池组容量的判断，如果额定容量不合理，则不进行SOH计算
            # 2025.3.23 增加对电池类型的判断，若电池类型不符合要求，则不进行SOH计算
            if full_cap == '/' or full_cap < 1 or full_cap > 600 or  (battery_type != 'LFP' and battery_type != 'NCM') or cur_total_error > 6 or dec_soc <= 30 :
                add_out_dir_info(rlt_res, '动力电池组容量保有率评分', 'N/A', '', '')
                add_out_dir_info(rlt_res, '动力电池组剩余容量', 'N/A', '', '')
                # add_out_dir_info(rlt_res, '动力电池组剩余总能量', 'N/A', '', '')
                # add_out_dir_info(rlt_res, '动力电池能量损失速率', 'N/A', '', '')
                add_out_dir_info(rlt_res, '可行驶里程估算', [0, 0], '', '')
                return rlt_res

            # 统计计算值的soh
            alg_soh_adj = 0
            alg_soh = kwargs['alg_ah']*100/full_cap
            # 2025.4.18 增加充电桩充电电流对SOH的校准
            # soc区间大于50时，alg_soh结果有效
            if dec_soc >30:
                if alg_soh < 90 and kwargs['charge_efficiency'] > 95 and kwargs['charge_efficiency'] < 100:
                    alg_soh_adj = alg_soh*100/ kwargs['charge_efficiency']
                else:
                    alg_soh_adj = alg_soh
            
            # 计算最高节电芯SOH值, 增加对充电工况的限制
            # max_volt_soh = 'N/A'
            # if dec_soc > 40 and (kwargs['soc_start'] <50 or kwargs['soc_end'] > 96)  and  df_cleaned['vol_max'].max() > df_cleaned['vol_max'].mode()[0]:
            if dec_soc > 30 and df_cleaned['vol_max'].max() > df_cleaned['vol_max'].mode()[0]:
                try:
                    soh_output = run_soh(data_piece, frequency, full_cap, battery_type, all_volt_names, so_path, cells_for_ods,
                                        sup_out_url='', save_flag=0)
                    # 动力电池组剩余容量判定结果
                    # best_std = np.argmax(soh_output.soh_std_list)
                    # elect_soh = soh_output.soh_list[best_std]  #min(soh_output.soh_list)
                    elect_soh = soh_output.soh_list[0]

                    # 2025.01.03 增加soh计算loss判断，去掉这个值的判断
                    # 电化学计算值比安时积分大
                    if  elect_soh > alg_soh_adj and elect_soh < 100:  
                        # max_volt_soh = round(elect_soh,2)
                        # if max_volt_soh  > 100:
                        #    max_volt_soh = 'N/A'
                        # else:
                        alg_soh_adj = round(elect_soh,2)

                    for i in range(len(soh_output.cell_list)):
                        add_out_dir_info(rlt_res, 'soh_'+ soh_output.cell_list[i], round(soh_output.soh_list[i],2), '', '')
                        add_out_dir_info(rlt_res, 'remain_'+ soh_output.cell_list[i], round(soh_output.soh_list[i]*full_cap/100, 2), '', '')  
                        add_out_dir_info(rlt_res, 'std_'+ soh_output.cell_list[i], soh_output.soh_std_list[i], '', '')
                # 如果电化学计算报错
                except Exception as e:
                    log.logger.error(f"electrochem soh error: {traceback.print_exc()}")
            
            soh_for_judge = alg_soh_adj   # 电池组平均SOH值
            # 2024.11.29 测试出SOH百分比会大于100%，需要修正
            if soh_for_judge > 103 or soh_for_judge < 10:
                add_out_dir_info(rlt_res, '动力电池组容量保有率评分', 'N/A', '', '')
                add_out_dir_info(rlt_res, '动力电池组剩余容量', 'N/A', '', '')
                add_out_dir_info(rlt_res, '可行驶里程估算', ['N/A', 'N/A'], '', '')
                return rlt_res
            elif soh_for_judge >100:
                soh_for_judge = 99.90
            
            # avga = df_cleaned['vol_total'].mean()
            
            if kwargs['rate_total_vol'] != '/' and kwargs['rate_total_vol']!= 'N/A' and kwargs['rate_total_vol'] >0:
                rate_total_vol = kwargs['rate_total_vol']
            else:
                rate_total_vol = df_cleaned['vol_total'].mean()  # 平均电压
         
            soh_value = round(soh_for_judge,2)

            # 2025.4.2 增加一致性容量损失判断, 最低单体电压上传，且极差值在合理范围, 且必须为恒流段 ---- 非恒流段初略评估
            dec_volt_soh = '/'
            if battery_type == 'LFP':
                limit_v = 0.35
            else:
                limit_v = 0.2
            
            if kwargs['min_volt'] != '/':
                
                dec_volt =  kwargs['max_volt'] - kwargs['min_volt']
                # if  dec_volt > limit_v and dec_volt < 1.5 and max_volt_soh != 'N/A' and max_volt_soh > soh_for_judge: 
                #     dec_volt_soh =  round(max_volt_soh - soh_for_judge,2)

                    # # 计算差值的绝对值
                    # data['abs_diff'] = abs(data['vol_max'] - kwargs['min_volt'])
                    # # 找到绝对值差最小的行，并按倒序返回（即返回最后一行）
                    # min_diff_row = data.loc[data['abs_diff'] == data['abs_diff'].min()].iloc[-1]
                    # # 判断剩下的充电段是否为恒流段：
                    # filtered_current = data[data['date_time'] > min_diff_row['date_time']]['current']
                    # cur_dec = abs(filtered_current.max() - filtered_current.min())
                    # if cur_dec < full_cap*0.05:
                    #     min_volt_soh = round(min_diff_row['bms_ah']*100/full_cap,2)
                    #     dec_volt_soh = round(soh_for_judge - min_volt_soh, 2)
                    # # 非恒流则粗略评估
                    # else:
                    #     cur_line = filtered_current.mean()/filtered_current.max()    # 初步矫正电流系数
                    #     min_volt_soh = round(soh_for_judge*cur_line,2)
                    #     dec_volt_soh = round(soh_for_judge - min_volt_soh, 2)
            
            if soh_value < 75:
                # if dec_volt_soh != '/':
                #     summary = f'当前电池组剩余容量（SoH）：{str(soh_value)}%'
                # else:    
                summary = f'当前电池组剩余容量百分比（SOH）：{str(soh_value)}%, 容量损失百分比为{round((100 - soh_value),2)}%，电池容量健康度低于正常（>75%）范围。若车辆电池使用年限大于5年或行驶里程大于10万公里，属于正常容量衰减'
                advice = f"【电池组剩余容量】"
            else:
                # if dec_volt_soh != '/':
                #     summary = f'当前电池组剩余容量（SoH）：{str(soh_value)}%'  # f'最高单体电芯剩余容量SoH值为：{str(max_volt_soh)}%，电池组剩余容量SoH值为：{str(soh_value)}%， 电压一致性容量损失值预估为：{dec_volt_soh}%。'
                # else:
                summary = f'当前电池组剩余容量百分比（SOH）：{str(soh_value)}%，剩余容量合格'

            dec_kwh = '/'
            if len(kwargs) > 0 and kwargs['year'] > 0:  # 传入电车的使用年限
                dec_kwh = round((100 - soh_for_judge)*full_cap* rate_total_vol/100000/kwargs['year'],2)    # 动力电池能量损失速率

            kwh_for_judge = soh_for_judge * full_cap * rate_total_vol/100000
            kw_total_min = round(kwh_for_judge*5,2)
            kw_total_max = round(kwh_for_judge*7,2)
            add_out_dir_info(rlt_res, '动力电池组容量保有率评分', round(soh_for_judge,2), summary, advice)
            add_out_dir_info(rlt_res, '动力电池组剩余容量', round(soh_value*full_cap/100, 2), '', '')
            add_out_dir_info(rlt_res, '动力电池组剩余总能量', round(kwh_for_judge,2), '', '')
            add_out_dir_info(rlt_res, '动力电池能量损失速率', dec_kwh, '', '')
            add_out_dir_info(rlt_res, '可行驶里程估算', [kw_total_min, kw_total_max], '', '')
            add_out_dir_info(rlt_res, '电化学运行时长(s)', round(time.time()- st,2), '', '')
            return rlt_res
    except Exception as e:
        log.logger.error(f"soh error: {traceback.print_exc()}")
        add_out_dir_info(rlt_res, '动力电池组容量保有率评分', 'N/A', '', '')
        # add_out_dir_info(rlt_res, '动力电池组容量保有率评分', 100, '', '')
        add_out_dir_info(rlt_res, '动力电池组剩余容量', 'N/A', '', '')
        # add_out_dir_info(rlt_res, '动力电池组剩余总能量', 'N/A', '', '')
        # add_out_dir_info(rlt_res, '动力电池能量损失速率', 'N/A', '', '')
        add_out_dir_info(rlt_res, '可行驶里程估算', ['N/A', 'N/A'], '', '')
        return rlt_res

    

