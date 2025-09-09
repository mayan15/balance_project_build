# -*- coding: utf-8 -*-

import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 移动平均函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def set_edge():
    # 边框设置
    bwith = 2 # 边框宽度设置为2
    edge = plt.gca() # 获取边框
    edge.spines['left'].set_color('black') # 设置左‘脊梁’为黑色
    edge.spines['bottom'].set_color('black')  
    edge.spines['right'].set_color('black')  
    edge.spines['top'].set_color('black')
    edge.spines['bottom'].set_linewidth(bwith)
    edge.spines['left'].set_linewidth(bwith)
    edge.spines['top'].set_linewidth(bwith)
    edge.spines['right'].set_linewidth(bwith)

def set_margin(margin):
    plt.gcf().subplots_adjust(left=margin[0], top=margin[1], right=margin[2], bottom=margin[3], hspace=0.3, wspace=None)  # 调整画布边界，避免出现坐标名称被截断的情况

def Text(text, position, fontSize):
    edge = plt.gca()
    plt.text(position[0], position[1], text, fontsize=fontSize, color='#23272C', \
             transform=edge.transAxes, verticalalignment='bottom', horizontalalignment='center') 


def run_dqdv_consis(df_cleaned, log):
    """
            dq/dv : 容量对电压一阶导

            :param df_cleaned:  数据清洗算法输出的df数据序列
            :param data_overview_rlt: 数据清洗算法输出的统计结果或关键参数值
            :param picture_save_path: 数据画像生成的图片保存路径
            :param pickle_save_path:  数据处理结果，用于其他程序读取
            :return:   rlt_res = {
                                "code_id": 20,
                                "describe": "dv/dq",
                                "out": {},
                                "summary": [],
                                "table": [],
                                "ErrorCode": [0, 0, '']}，

    """
    try:
        st = time.time()
        result_dict = {}
        vol_max = df_cleaned['vol_max']
        vol_mid = df_cleaned['vol_mid']
        vol_total = df_cleaned['vol_total']

        QAh = df_cleaned['bms_ah']
        soc = df_cleaned['soc']

        new_point_start_vol_max = vol_max[0]
        new_point_start_vol_mid = vol_mid[0]
        new_point_start_vol_total = vol_total[0]
        new_point_start_q = 0 #QAh[0]

        dtQ =  round(QAh.iloc[-1]/15, 1)        # 15个point 

        maxvol_dv_dq_rlt = []
        midvol_dv_dq_rlt = []
        totalvol_dv_dq_rlt = []
        dsoc_dv_rlt = []

        for i in range(len(vol_max)):
            temp = {'index': 0, 'total_dec_vol': 0, 'max_dec_vol': 0, 'mid_dec_vol':0, 'dq': 0, 'max_dvdq': 0, 'total_dvdq': 0, 'mid_dvdq' : 0,'soc': 0}
            if abs(QAh[i] - new_point_start_q) > dtQ:
                temp['index'] = i
                temp['total_dec_vol'] = abs(vol_total[i] - new_point_start_vol_total)*1000  # convert to mv
                temp['max_dec_vol'] = abs(vol_max[i] - new_point_start_vol_max)*1000
                temp['mid_dec_vol'] = abs(vol_mid[i] - new_point_start_vol_mid)*1000

                temp['dq'] = abs(QAh[i] - new_point_start_q)
                temp['soc'] = soc[i]
                temp['total_dvdq'] = round(temp['total_dec_vol']/temp['dq'],3)
                temp['max_dvdq'] = round(temp['max_dec_vol']/temp['dq'],3)
                temp['mid_dvdq'] = round(temp['mid_dec_vol']/temp['dq'],3)
                
                new_point_start_vol_total = vol_total[i]
                new_point_start_vol_max = vol_max[i]
                new_point_start_vol_mid = vol_mid[i]
                new_point_start_q = QAh[i]

                maxvol_dv_dq_rlt.append(temp['max_dvdq'])
                midvol_dv_dq_rlt.append(temp['mid_dvdq'])
                totalvol_dv_dq_rlt.append(temp['total_dvdq'])
                dsoc_dv_rlt.append(temp['soc'])

        diff_dq_dv = np.array(maxvol_dv_dq_rlt) - np.array(midvol_dv_dq_rlt)
        diff_dq_dv_mean = abs(np.max(diff_dq_dv))
        

        if diff_dq_dv_mean < 20: 
            dqdv_score =  round(100 - diff_dq_dv_mean*2, 2)     # 1 mv/Ah 差距，分值1， 修改一下分数计算 
        else:
            dqdv_score = 40

        summary =[]
        advice =[]
        if dqdv_score< 70:
            summary.append(f'电池组单位时间内容压比值：{diff_dq_dv_mean:.2f} mv/Ah，数值高。')
            # advice.append(f'电池组单位时间内容压比值：{diff_dq_dv_mean:.2f} mv/Ah, 正常值小于20，建议检查电车续航里程显示是否准确。')
        elif dqdv_score < 85:
            summary.append(f'电池组单位时间内容压比值：{diff_dq_dv_mean:.2f} mv/Ah，数值偏高。')
            # advice.append('电池组容差比值较高，建议关注。')
        else:
            summary.append(f'电池组单位时间内容压比值：{diff_dq_dv_mean:.2f} mv/Ah，数值正常。')
        
        result_dict['error'] = 0
        result_dict['class'] = 'consistency'
        result_dict['name'] = 'dq/dv'
        result_dict['score'] =  dqdv_score
        result_dict['summary'] = summary
        result_dict['advice'] = advice

        # ''' 对结果值进行画图 '''
        # y1 = np.array(maxvol_dv_dq_rlt)
        # y2 = np.array(midvol_dv_dq_rlt)
        # # y3 = np.array(totalvol_dv_dq_rlt)
        # x1 = np.array(dsoc_dv_rlt)
        # # 线性插值
        # linear_interp1 = interp1d(x1, y1, kind='linear', fill_value='extrapolate')
        # x1_new = np.linspace(x1.min(), x1.max(), num=100)  # 生成新的x值
        # y1_linear = linear_interp1(x1_new)

        # linear_interp2 = interp1d(x1, y2, kind='linear', fill_value='extrapolate')
        # # x2_new = np.linspace(x2.min(), x2.max(), num=500)  # 生成新的x值
        # y2_linear = linear_interp2(x1_new)

        # # linear_interp3 = interp1d(x3, y1, kind='linear', fill_value='extrapolate')
        # # x3_new = np.linspace(x3.min(), x3.max(), num=500)  # 生成新的x值
        # # y3_linear = linear_interp2(x3_new)
       
        # # 设置窗口大小
        # window_size = 50  # 可根据需要调整窗口大小
        # y1_smoothed = moving_average(y1_linear, window_size)
        # x1_smoothed = x1_new[window_size-1:]  
        # y2_smoothed = moving_average(y2_linear, window_size)
        # x2_smoothed = x1_new[window_size-1:]  
        # # y3_smoothed = moving_average(y3_linear, window_size)
        # # x3_smoothed = x3_new[window_size-1:] 
        # # 绘制平滑曲线
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体  
        # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题 
        # plt.figure(figsize=(18, 6))
        # set_edge()
        # set_margin([0.1, 0.95, 0.95, 0.15])
        # plt.title("dv/dq", fontsize=22)

        # plt.plot(x1_smoothed, y1_smoothed,  label='max_vol', color='r' )
        # plt.plot(x2_smoothed, y2_smoothed,  label='mid_vol', color='g')
        # #plt.plot(x3_smoothed, y3_smoothed,  label='total_vol', color='b')

        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # plt.ylabel('dv/dq',fontsize=20)
        # plt.xlabel('SOC (%)',fontsize=20)
        # plt.xticks(fontsize=22)
        # plt.yticks(fontsize=22)
        # plt.legend(loc='upper right',fontsize=22, bbox_to_anchor=(0.85, 1))

        # Text(f'图{7}: dv/dq曲线变化', [0.5, -0.20], 20)
        # plt.savefig(f'./dq_smooth.png')
        # plt.tight_layout()
        
        log.logger.debug(f"Consist temp calculate time: {round(time.time()-st,2)} seconds")

        return result_dict

    except Exception as e:
        log.logger.error(f"dq/dv consis error: {traceback.print_exc()}")
        result_dict['error'] = 99
        result_dict['score'] = 100
        result_dict['summary'] = []#['数据不足，容压比一致性暂无法评估']
        result_dict['advice'] = []#['数据不足，容压比一致性相关暂无建议']
        return result_dict
