import pandas as pd
import numpy as np
import time
import traceback


def cal_vols_noise(df):
    """ 
    计算电压噪声
    """
    filtered_volts = df.filter(like='vol_max').join(df.filter(like='vol_mid'))
    row_rms = round(filtered_volts.std(ddof=0),3) 
    # row_rms = round(filtered_volts.std(),3)
    row_avg = round(filtered_volts.mean(),3)
    row_ppn = round(filtered_volts.max() - filtered_volts.min(),3)  # 峰值噪声
    row_snr = round(20*np.log10(row_avg/row_rms),3)   # 信噪比
    # 将平均值和标准差整合到一个新的DataFrame中
    result = pd.DataFrame({
        "RMS": row_rms,
        "PPN": row_ppn,     # 峰值噪声
        "SNR": row_snr      # 噪声比
    })
    
    # 计算所有列的整体平均值和标准差
    overall_RMS_MAX = result['RMS'].max()
    max_RMS_col_name = result['RMS'].idxmax()

    overall_PPN_MAX = result['PPN'].max()
    max_PPN_col_name = result['PPN'].idxmax()

    overall_SNR = result['SNR'].min()                # 越大越好
    max_SNR_col_name = result['SNR'].idxmin()
  
    # 创建当天的结果 DataFrame
    result_summary = pd.DataFrame({
        # 'date': [date],
        'overall_RMS_MAX': [overall_RMS_MAX],
        'max_RMS_col_name': [max_RMS_col_name],
        
        'overall_PPN_MAX': [overall_PPN_MAX],
        'max_PPN_col_name': [max_PPN_col_name],
        
        'overall_SNR': [overall_SNR],
        'max_SNR_col_name': [max_SNR_col_name]
    })  
    return result_summary

def sigmoid_normalize_x(x, mu=1, s=5):
    return 1 / (1 + np.exp(-(x - mu) / s))

def run_vol_sensor_check(df, battery_type, log):
    """

    """
    try:
        result_dict = {}
        st = time.time()
        vol_sensor_value_correct_score = 'N/A' 
        vol_correlation_score = 'N/A' 
        
        # ''' 计算电压传感器有效性判断和建议 '''
        summary = []
        advice = []

        # 20250731 新增总电压数据上传正常判断

        diff_cur = df['current'].diff().dropna()
       
        diff_bms_total_vol = df['vol_total'].diff().dropna()
        vol_len = len(diff_bms_total_vol)
        condition = (diff_bms_total_vol< -9) & (diff_cur > -50)
        indices = condition.sum()
        indices_0 = diff_bms_total_vol.loc[diff_bms_total_vol == 0].index

        diff_total_vol = df['charge_out_vol'].diff().dropna()
        condition = (diff_total_vol<-9) & (diff_cur > -50)
        indices_2 = condition.sum()
        indices_1 = diff_total_vol.loc[diff_total_vol == 0].index

        if  df['vol_total'].max() > df['vol_total'].mode()[0] and df['charge_out_vol'].max() > df['charge_out_vol'].mode()[0] and indices<5 and indices_2<5 and len(indices_0)<vol_len and len(indices_1)<vol_len:
            # 计算电压传感器有效性判断和建议  (只判断电压传感器有效性，根据总电压来判断,非单体电压进行判断)
            # 计算BMS总电压与充电桩总电压相关系（两者理论是强相关）
            correlation_vol_max = round(np.corrcoef(df['vol_total'], df['charge_out_vol'])[0, 1],3)
            vol_correlation_score = round(100* correlation_vol_max,2)
            
            # 计算BMS总电压与充电桩总电压的绝对误差和相对误差
            # vol_rule = round(0.04* df['charge_out_vol'].mean(),2)
            vol_rule = 18    # 换成一样的分母 --- 差别： 18V

            vol_total_error = round(abs((df['vol_total'] - df['charge_out_vol']).mean()),2)
            vol_total_relative_error = round(vol_rule/vol_total_error ,2) 

            vol_total_relative_normalize = round(sigmoid_normalize_x(vol_total_relative_error),2) 
            vol_sensor_value_correct_score = round(vol_total_relative_normalize*100,2)
                
            if vol_sensor_value_correct_score < 50:
                # 充电过程中电芯间电压传感器测量有效性系数
                # summary.append(f'电压传感器测量有效性指标值{vol_total_relative_normalize}，低于车辆BMS电压传感器测量有效性平均值{round(1/vol_total_relative_error,2)}倍， 该值越低，表明电压传感器测量值与真实值之间偏差越大。')
                summary.append(f'BMS上传总电压数值与充电桩总电压数值存在偏差，平均相差{vol_total_error}V，高于正常（<{vol_rule}V）范围')
                # advice.append(f'电池包电压传感器有效性指标值{round(vol_sensor_value_correct_score/100,2)}，正常范围0.6-1.0，疑似存在BMS电压检测误差问题，建议检查车载BMS系统。')
                advice.append(f'【BMS总电压测量数值】')
            else:
                summary.append(f'总电压测量数值合格')
        else:
            pass 
            # summary.append('数据不支持，电压传感器有效性暂无法评估。')

        result_dict['error'] = 0
        result_dict['class'] = 'sensor'
        result_dict['name'] = 'votage_sensor_check'
        result_dict['score'] = [vol_sensor_value_correct_score, vol_correlation_score]
        result_dict['summary'] = summary
        result_dict['advice'] = advice
    
        log.logger.debug(f"voltage sensor check calculate time: {round(time.time()-st,2)} seconds")
        return result_dict
    except Exception as e:
        log.logger.error(f"voltage sensor check error: {traceback.print_exc()}")
        result_dict['error'] = 99
        result_dict['score'] = ['N/A', 'N/A']
        result_dict['summary'] = [] #['数据不足，无法进行BMS电压传感器分析。']
        result_dict['advice'] = []
        return result_dict
       
