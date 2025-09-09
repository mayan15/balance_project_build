import numpy as np
import pandas as pd
import traceback
import pickle

from .consis_volt import run_volt_consis
from .consis_dq   import run_dqdv_consis
from .consis_temp import run_temp_consis
from .consis_soc  import run_soc_consis

from tool_utils.pvlog import Logger
from tool_utils.plot  import set_plot_properties

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


level = 'error'
log_consis = Logger('./logs/consistency.log', level= level)

def add_out_dir_info(rlt_res, key, value, confidence, explanation) :
    """
    向 rlt_res 中添加 key value and confidence explanation

    参数：
    rlt_res: 结果字典
    key: '充电起始总电压'或'充电结束总电压'
    value: 总电压值
    confidence:  评分/或者对评分的解释 
    explanation : 
    """
    # 初始化指定的键
    rlt_res['out'][key] = []
    rlt_res['out'][key].append(value)
    rlt_res['out'][key].append(confidence)
    rlt_res['out'][key].append(explanation)

@set_plot_properties(border_padding=0.15, figsize=(19, 6))
def plot_double_yaxis_pic(ax, ax1_y_value, ax1_y_color, ax1_lable,ax2_y_value, ax2_y_color, ax2_lable,ax1_x_value, ax_name, ylim_limit, x_indices, ax_fontsize, ax_legend_msg, picture_name, picture_save_path, pic_typ, timesign_type=True):
    try:
        indices = np.linspace(0, len(ax1_x_value) - 1, x_indices, dtype=int)
        indices_time = ax1_x_value.to_numpy()   
        time_x = ax1_x_value.iloc[indices].values

        for index, value in enumerate(ax1_y_value):
            if pic_typ[0] == 'plot':
                ax.plot(indices_time, value.to_numpy(), label= ax1_lable[index], color= ax1_y_color[index])
            elif pic_typ[0] == 'scatter':
                ax.scatter(indices_time, value.to_numpy(), label= ax1_lable[index], color= ax1_y_color[index])     

        ax.set_ylim(ylim_limit[0][0], ylim_limit[0][1])
        ax.tick_params(axis='y', labelsize=ax_fontsize)  # 设置y 轴 标签字体大小
        ax.set_xticks(time_x)               
        ax.set_xticklabels(time_x)
        ax.set_xticklabels(ax.get_xticks(), fontsize= ax_fontsize)
        ax.set_ylabel(ax_name[0], fontsize= ax_fontsize)
        ax.legend( loc=ax_legend_msg[0][0], fontsize=ax_fontsize, bbox_to_anchor=ax_legend_msg[0][1], frameon=False)  
        
        # 创建次要坐标轴
        ax2 = ax.twinx()  
        for index, value in enumerate(ax2_y_value):
            if pic_typ[1] == 'plot':
                ax2.plot(indices_time, value.to_numpy(), label= ax2_lable[index], color= ax2_y_color[index], linestyle='--')
            elif  pic_typ[1] == 'scatter':
                ax2.scatter(indices_time, value.to_numpy(), label= ax2_lable[index], color= ax2_y_color[index])     

        ax2.set_ylim(ylim_limit[1][0], ylim_limit[1][1])
        ax2.tick_params(axis='y', labelsize=ax_fontsize)   # 设置y 轴 标签字体大小
        ax2.set_ylabel(ax_name[1] , fontsize=ax_fontsize)  # 次纵坐标标签
        ax2.legend( loc= ax_legend_msg[1][0], fontsize=ax_fontsize, bbox_to_anchor=ax_legend_msg[1][1],frameon=False)  
    
        # 设置横坐标（x）时间显示的点
        ax2.set_xticks(time_x)               
        ax2.set_xticklabels(time_x)
        if timesign_type:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 显示 mm-dd hh 格式

        # 设置横坐标的主要刻度
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
        plt.text(0.5, 1.0, picture_name, fontsize=24, color='#23272C', \
                transform= ax.transAxes, verticalalignment='bottom', horizontalalignment='center') 
        plt.savefig(f"{picture_save_path}/{picture_name}.png")

    except Exception as e:
        log_consis.logger.error(f"plot_data error: {traceback.print_exc()}")


def run(df_raw, battery_type, max_voltage, min_voltage, max_temp, min_temp, picture_save_path, pickle_save_path,soh_rlt,data_cleaned=None):
    """
       
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
        if min_voltage == '/':            
            log_consis.logger.error("No minimum voltage data, set it to mean value: {}".format(min_voltage))

        vol_rlt  = run_volt_consis(df_raw, log_consis, max_voltage, min_voltage, soh_rlt, data_cleaned)
        temp_rlt = run_temp_consis(df_raw, log_consis, max_temp, min_temp, data_cleaned)
        dvdq_rlt = run_dqdv_consis(df_raw, log_consis)
        soc_rlt  = run_soc_consis(df_raw, log_consis, max_voltage, min_voltage, soh_rlt, data_cleaned)
       
        # vol_consistency_score = 'N/A'
        # temp_consistency_score = 'N/A'

        # if vol_rlt['score'][0] != 'N/A':
        #     vol_consistency_score = round(vol_rlt['score'][0],2)
        # if temp_rlt['score'][0] != 'N/A':    
        #     temp_consistency_score = round(temp_rlt['score'][0],2)  
        
        if vol_rlt['score'][0] != 'N/A' and temp_rlt['score'][0] != 'N/A' and soc_rlt['score'][0] != 'N/A':
            consistency_socre = round(vol_rlt['score'][0]*0.4 + temp_rlt['score'][0]*0.4 + soc_rlt['score'][0]*0.2,2)
        elif vol_rlt['score'][0] != 'N/A' and temp_rlt['score'][0] != 'N/A':
            consistency_socre = round((vol_rlt['score'][0]*0.4 + temp_rlt['score'][0]*0.4)/0.8,2)
        elif vol_rlt['score'][0] != 'N/A' and soc_rlt['score'][0] != 'N/A':
            consistency_socre = round((vol_rlt['score'][0]*0.4 + soc_rlt['score'][0]*0.2)/0.6,2)
        elif soc_rlt['score'][0] != 'N/A' and temp_rlt['score'][0] != 'N/A':
            consistency_socre = round((soc_rlt['score'][0]*0.2 + temp_rlt['score'][0]*0.4)/0.6,2)
        # elif vol_rlt['score'][0] != 'N/A':
        #     consistency_socre = round(vol_rlt['score'][0],2)
        # elif temp_rlt['score'][0] != 'N/A':
        #     consistency_socre = round(temp_rlt['score'][0],2)
        # elif soc_rlt['score'][0] != 'N/A':
        #     consistency_socre = round(soc_rlt['score'][0],2)
        else:
            consistency_socre = 'N/A'

        # log.logger.debug(f"vol_rlt: {vol_rlt}, temp_rlt: {temp_rlt}, dvdq_rlt: {dvdq_rlt}")
        # 一致性总评结果输出
        add_out_dir_info(rlt_res, '电池组一致性总评分', consistency_socre, '', '')
        add_out_dir_info(rlt_res, '电池组电压极差', vol_rlt['score'][2], '', '')    
        add_out_dir_info(rlt_res, '电池组电压一致性评分', vol_rlt['score'][0], vol_rlt['summary'], vol_rlt['advice'])
        add_out_dir_info(rlt_res, '电池组温度一致性评分', temp_rlt['score'][0], temp_rlt['summary'], temp_rlt['advice'])
        add_out_dir_info(rlt_res, '电池组SOC一致性评分', soc_rlt['score'][0], soc_rlt['summary'], soc_rlt['advice'])
        add_out_dir_info(rlt_res, '电池组容压比一致性评分', dvdq_rlt['score'], dvdq_rlt['summary'], dvdq_rlt['advice'])

        # --------out picture ---------#
        # plot_vol_current(df_raw, 18 ,"图1-1:充电电压-电流曲线", picture_save_path)
        # plot_vol_number(df_raw, 18 ,"图1-2:最高单体电压-电芯节号分布图", picture_save_path)
        # 图 1
        # plot_double_yaxis_pic([df_raw['vol_max'], df_raw['vol_mid']], 
        #                       ['red','green'], 
        #                       ['max_vol', 'mid_vol'], 
        #                       [df_raw['current']],
        #                       ['orangered'],
        #                       ['current'],
        #                       df_raw['date'], 
        #                       ['单体电压(V)', '电流(A)'], 
        #                       [(min(df_raw['vol_max'])*0.9, max(df_raw['vol_max'])*1.1), (min(df_raw['current'])*0.9, max(df_raw['current'])*1.1)], 
        #                       10, 
        #                       18,
        #                       [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
        #                       '图1-1:充电单体电压-电流充电曲线', 
        #                       picture_save_path,
        #                       ['plot', 'plot']
        #                       )
        # 图 2
        plot_double_yaxis_pic([df_raw['vol_max']], 
                              ['red'], 
                              ['max_vol'], 
                              [df_raw['max_vol_no']],
                              ['blue'],
                              ['cell_no'],
                              df_raw['date'], 
                              ['单体电压(V)', '电池节号'], 
                              [(min(df_raw['vol_max'])*0.9, max(df_raw['vol_max'])*1.1), (min(df_raw['max_vol_no'])-5, max(df_raw['max_vol_no'])+5)], 
                              10, 
                              18,
                              [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
                              '图1-2:最高单体电压-电池节号分布', 
                              picture_save_path,
                              ['plot', 'scatter']
                              )
        
        # 图 3
        plot_double_yaxis_pic([df_raw['temp_max'], df_raw['temp_min']], 
                              ['red','green'], 
                              ['max_tep', 'min_tep'], 
                              [df_raw['current']],
                              ['orangered'],
                              ['current'],
                              df_raw['date'], 
                              ['温度(℃)', '电流(A)'], 
                              [(min(df_raw['temp_max'])*0.9, max(df_raw['temp_max'])*1.1), (min(df_raw['current'])*0.9, max(df_raw['current'])*1.1)], 
                              10, 
                              18,
                              [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
                              '图1-3:充电温度-电流曲线', 
                              picture_save_path,
                              ['plot', 'plot']
                              )
        # # 图 4
        # plot_double_yaxis_pic([df_raw['temp_max']], 
        #                       ['red'], 
        #                       ['max_tep'], 
        #                       [df_raw['max_temp_no']],
        #                       ['blue'],
        #                       ['temp_no'],
        #                       df_raw['date'], 
        #                       ['温度(℃)', '温度探头号'], 
        #                       [(min(df_raw['temp_max'])*0.9, max(df_raw['temp_max'])*1.1), (min(df_raw['max_temp_no'])-5, max(df_raw['max_temp_no'])+5)], 
        #                       10, 
        #                       18,
        #                       [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
        #                       '图1-4:最高温度-探头号分布', 
        #                       picture_save_path,
        #                       ['plot', 'scatter']
        #                       )
        # # 图 5
        # plot_double_yaxis_pic([df_raw['temp_min']], 
        #                       ['red'], 
        #                       ['min_tep'], 
        #                       [df_raw['min_temp_no']],
        #                       ['blue'],
        #                       ['temp_no'],
        #                       df_raw['date'], 
        #                       ['温度(℃)', '温度探头号'], 
        #                       [(min(df_raw['temp_min'])*0.9, max(df_raw['temp_min'])*1.1), (min(df_raw['min_temp_no'])-5, max(df_raw['min_temp_no'])+5)], 
        #                       10, 
        #                       18,
        #                       [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
        #                       '图1-5:最低温度-探头号分布', 
        #                       picture_save_path,
        #                       ['plot', 'scatter']
        #                       )
        # 图 6
        plot_double_yaxis_pic([df_raw['vol_total']], 
                              ['red'], 
                              ['tvol'], 
                              [df_raw['soc']],
                              ['orangered'],
                              ['soc'],
                              df_raw['date'], 
                              ['总电压(V)', 'SOC(%)'], 
                              [(min(df_raw['vol_total'])*0.9, max(df_raw['vol_total'])*1.1), (min(df_raw['soc'])-5, max(df_raw['soc'])+5)], 
                              10, 
                              18,
                              [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
                              '图1-6:充电总电压-SOC变化曲线', 
                              picture_save_path,
                              ['plot', 'plot']
                              )

        
        # 图 7
        plot_double_yaxis_pic([df_raw['vol_max']], 
                              ['red'], 
                              ['max_vol'], 
                              [df_raw['soc']],
                              ['orangered'],
                              ['soc'],
                              df_raw['date'], 
                              ['最高单体电压(V)', 'SOC(%)'], 
                              [(min(df_raw['vol_max'])*0.9, max(df_raw['vol_max'])*1.1), (min(df_raw['soc'])-5, max(df_raw['soc'])+5)], 
                              10, 
                              18,
                              [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
                              '图1-7:充电最高单体电压-SOC变化曲线', 
                              picture_save_path,
                              ['plot', 'plot']
                              )
        
        # 图 8
        # plot_double_yaxis_pic([df_raw['bms_ah'][1:]], 
        #                       ['red'], 
        #                       ['ah'], 
        #                       [df_raw['soc'][1:]],
        #                       ['orangered'],
        #                       ['soc'],
        #                       df_raw['date'][1:], 
        #                       ['充电容量(Ah)', 'SOC(%)'], 
        #                       [(min(df_raw['bms_ah'][1:])*0.9, max(df_raw['bms_ah'][1:])*1.1), (min(df_raw['soc'])-5, max(df_raw['soc'])+5)], 
        #                       10, 
        #                       18,
        #                       [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
        #                       '图1-8:充电容量-SOC变化曲线', 
        #                       picture_save_path,
        #                       ['plot', 'plot']
        #                       )
        
        # # 图 9
        # plot_double_yaxis_pic([df_raw['bms_kwh'][1:]], 
        #                       ['green'], 
        #                       ['KW·h'], 
        #                       [df_raw['soc'][1:]],
        #                       ['orangered'],
        #                       ['soc'],
        #                       df_raw['date'][1:], 
        #                       ['充电电量(KW·h)', 'SOC(%)'], 
        #                       [(min(df_raw['bms_kwh'][1:])*0.9, max(df_raw['bms_kwh'][1:])*1.1), (min(df_raw['soc'])-5, max(df_raw['soc'])+5)], 
        #                       10, 
        #                       18,
        #                       [['upper right',(1.2,0.5)], ['upper right',(1.2,0.7)]], 
        #                       '图1-9:充电电量-SOC变化曲线', 
        #                       picture_save_path,
        #                       ['plot', 'plot']
        #                       )
        
        # --------out pickle ---------#
        # with open(pickle_save_path + '/consistency.pkl', "wb") as f:
        #     pickle.dump(rlt_res, f)
        return rlt_res
    except Exception as e:
        log_consis.logger.error(f"consistency error: {traceback.print_exc()}")
        add_out_dir_info(rlt_res, '电池组一致性总评分', 'N/A' , '', '')
        add_out_dir_info(rlt_res, '电池组电压一致性评分', vol_rlt['score'][0], vol_rlt['summary'], vol_rlt['advice'])
        add_out_dir_info(rlt_res, '电池组温度一致性评分', temp_rlt['score'][0], temp_rlt['summary'], temp_rlt['advice'])
        add_out_dir_info(rlt_res, '电池组SOC一致性评分', soc_rlt['score'], soc_rlt['summary'], soc_rlt['advice'])
        add_out_dir_info(rlt_res, '电池组容压比一致性评分', dvdq_rlt['score'], dvdq_rlt['summary'], dvdq_rlt['advice'])
        return rlt_res