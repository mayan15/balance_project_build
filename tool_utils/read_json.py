import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import ast 

from vin import get_data_from_csv
from plot import plot_scatter_x_y

import seaborn as sns
from sklearn.cluster import KMeans

current_cwd = os.getcwd()    
# 假设JSON文件存储在某个目录下
json_dir = '/root/data/sp/charging_project_build/data/json/'
csv_dir_path = '/root/data/sp/charging_project_build/data/csv/vin2025020602_report.xlsx'

def  get_data_from_json(json_dir_path):
    """
    从JSON文件中读取数据，并返回为df
    """
    # 初始化一个空列表，用于存储所有JSON数据
    data_list = []
    # 遍历所有JSON文件, 并返回为df数组
    for filename in os.listdir(json_dir_path):
        try:
            if filename.endswith('.json'):
                with open(os.path.join(json_dir_path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 处理数据 --  将JSON字段中的关键信息提取出，并形成专门的列名
                    dir_data= {}
                    # 车辆基本信息
                    dir_data['VIN'] =  data["data"]["car_info"]["车辆识别码"]["结果值"]
                    dir_data['vehicle_manufacturerv'] =  data["data"]["car_info"]["车辆制造商"]["结果值"] 
                    dir_data['production_date'] =  data["data"]["car_info"]["生产日期"]["结果值"]
                    dir_data['file_name'] =  data["data"]["charging_management_system_info"]["充电桩SN号"]["结果值"]
                    dir_data['sn'] =  data["data"]["charging_management_system_info"]["充电桩SN号"]["结果值"][5:19]
                    # 电池基本信息
                    dir_data['cell_type'] =  data["data"]["battery_info"]["电池类型"]["结果值"]
                    dir_data['cell_capacity'] =  data["data"]["battery_info"]["额定容量"]["结果值"]
                    dir_data['battery_pack_number'] =  data["data"]["battery_info"]["电芯数量"]["结果值"]
                    dir_data['battery_pack_rate_volt'] =  data["data"]["battery_info"]["额定总压"]["结果值"]
                    dir_data['battery_pack_rate_energy'] =  data["data"]["battery_info"]["额定能量"]["结果值"]
                    # BMS信息
                    dir_data['bms_alw_max_vol'] =  data["data"]["battery_info"]["BMS允许最高单体电压"]["结果值"]
                    dir_data['bms_alw_max_cur'] =  data["data"]["battery_info"]["BMS允许最高电流"]["结果值"]
                    dir_data['bms_alw_tvol'] =  data["data"]["battery_info"]["BMS允许最高总电压"]["结果值"]
                    dir_data['bms_alw_temp'] =  data["data"]["battery_info"]["BMS允许最高温度"]["结果值"]
                    dir_data['charge_max_vol'] =  data["data"]["battery_info"]["充电输出最高电压"]["结果值"]
                    dir_data['charge_min_vol'] =  data["data"]["battery_info"]["充电输出最低电压"]["结果值"]
                    dir_data['charge_max_cur'] =  data["data"]["battery_info"]["充电输出最高电流"]["结果值"]
                    dir_data['charge_min_cur'] =  data["data"]["battery_info"]["充电输出最低电流"]["结果值"]   
                    # 充电基本信息
                    values = data["data"]["charging_info"]["充电SOC区间"]["结果值"].split('-')
                    dir_data['soc_start'] = float(values[0])
                    dir_data['soc_end'] =  float(values[1])

                    if data["data"]["voltage_info"]["蓄电池单体最大压差值"]["结果值"] != '/':
                        dir_data['max_cell_vol_dec'] = float(data["data"]["voltage_info"]["蓄电池单体最大压差值"]["结果值"])
                    else:
                        dir_data['max_cell_vol_dec'] = np.nan
                    
                    if data["data"]["temperature_info"]["温度探头检测最大差值"]["结果值"] != '/':
                        dir_data['max_temp_dec'] = float(data["data"]["temperature_info"]["温度探头检测最大差值"]["结果值"])
                    else:
                        dir_data['max_temp_dec'] = np.nan
                      
                    dir_data['charging_time'] =  float(data["data"]["charging_info"]["当次充电时间"]["结果值"])
                    dir_data['high_rate_time'] =  data["data"]["current_info"]["快充时长"]["结果值"]  
                    dir_data['charging_capacity'] =  data["data"]["charging_info"]["当次充电容量"]["结果值"]
                    dir_data['charging_kwh'] =  data["data"]["charging_info"]["当次充电度数"]["结果值"]

                    values = data["data"]["charging_info"]["充电总电压区间"]["结果值"].split('-')
                    dir_data['total_vol_start'] = float(values[0])
                    dir_data['total_vol_end'] =  float(values[1])

                    values = data["data"]["charging_info"]["充电电流区间"]["结果值"].split('-')
                    dir_data['current_min'] =  float(values[0])
                    dir_data['current_max'] =  float(values[1])
                    
                    # dir_data['current_mid'] =  data["data"]["battery_info"]["充电输出最低电流"]["结果值"]
                    values = data["data"]["charging_info"]["最大单体电压区间"]["结果值"].split('-')
                    dir_data['max_cell_vol_min'] =  float(values[0])
                    dir_data['max_cell_vol_max'] =  float(values[1])
                    values = data["data"]["charging_info"]["温度区间"]["结果值"].split('-')
                    dir_data['temp_min'] =  float(values[0])
                    dir_data['temp_max'] =  float(values[1])
                    

                    # 新增：充电数据详细统计信息 
                    dir_data['total_vol_dec_describe'] =  data["data"]["charging_info"]["充电桩输出电压与bms检查电压差值统计描述"]["结果值"]
                    dir_data['total_vol_dec_mid'] = round(json.loads(dir_data['total_vol_dec_describe'])[1],2)
                    dir_data['total_cur_dec_describe'] =  data["data"]["charging_info"]["充电桩输出电流与bms检查电流差值统计描述"]["结果值"]
                    dir_data['total_cur_dec_mid'] = round(json.loads(dir_data['total_cur_dec_describe'])[1],2)
                    
                    dir_data['vol_dec_mid_and_max_describe'] =  data["data"]["charging_info"]["充电单体最高与平均值差统计描述"]["结果值"]
                    dir_data['vol_total_describe'] =  data["data"]["charging_info"]["充电最高电压值统计描述"]["结果值"]
                    dir_data['current_describe'] =  data["data"]["charging_info"]["充电总电流值统计描述"]["结果值"]
                    dir_data['vol_max_describe'] =  data["data"]["charging_info"]["充电单体最高电压值统计描述"]["结果值"]

                    dir_data['temp_max_describe'] =  data["data"]["charging_info"]["充电最高温度值统计描述"]["结果值"]
                    dir_data['temp_min_describe'] =  data["data"]["charging_info"]["充电最低温度值统计描述"]["结果值"]
                    dir_data['charge_out_vol_describe'] =  data["data"]["charging_info"]["充电桩输出电压值统计描述"]["结果值"]
                    dir_data['charge_out_cur_describe'] =  data["data"]["charging_info"]["充电桩输出电流值统计描述"]["结果值"]
                    
                    
                    dir_data['max_vol_no_describe'] =  data["data"]["charging_info"]["充电单体节号分布描述"]["结果值"]
                    dir_data['max_temp_no_describe'] =  data["data"]["charging_info"]["充电最高温度探头号分布描述"]["结果值"]
                    dir_data['min_temp_no_describe'] =  data["data"]["charging_info"]["充电最低温度探头号分布描述"]["结果值"]
                    # 20240224新增-最高点温度探头变化特征、线阻值
                    dir_data['line_res'] =  round(float(data["data"]["charging_info"]["充电线阻值"]["结果值"]),4)
                    
                    # 处理温度变化特征值数据
                    max_temp_change_describe = ast.literal_eval(data["data"]["charging_info"]["最高温度变化分布描述"]["结果值"])
                    dir_data['max_temp_change_describe'] = max_temp_change_describe
                    datatime_list = max_temp_change_describe[0]
                    # 初始化一个空列表存储时间差（以秒为单位）
                    time_differences_in_seconds = []
                    import datetime
                    # 遍历列表，计算时间差
                    for i in range(1, len(datatime_list)):
                        time1 = datetime.datetime.strptime(datatime_list[i-1], '%Y-%m-%d %H:%M:%S')
                        time2 = datetime.datetime.strptime(datatime_list[i], '%Y-%m-%d %H:%M:%S')
                        # 计算时间差
                        difference = time2 - time1
                        # 将时间差转换为秒并添加到列表中
                        time_differences_in_seconds.append(int(difference.total_seconds()))  

                    dir_data['htemp_start_change_need_seconds'] = time_differences_in_seconds[0]
                    dir_data['htemp_start_change_need_seconds'] = time_differences_in_seconds[0]
                    dir_data['htemp_end_change_need_seconds'] = time_differences_in_seconds[-1]
                    dir_data['htemp_start_cur'] = max_temp_change_describe[1][1]
                    dir_data['htemp_end_cur'] = max_temp_change_describe[1][-1]
                    dir_data['htemp_median_cur'] = np.median(max_temp_change_describe[1])
                    dir_data['htemp_end_temp_dec'] =  max_temp_change_describe[5][-1] -  max_temp_change_describe[7][-1]                         
                    # 数据异常值比例
                    dir_data['abusive_current'] =  data["data"]["abusive_info"]["电流异常值比例"]["结果值"]
                    dir_data['abusive_vol'] =  data["data"]["abusive_info"]["电压值异常比例"]["结果值"]
                    dir_data['abusive_temp'] =  data["data"]["abusive_info"]["温度值异常比例"]["结果值"]
                    dir_data['cur_inconformity_rate'] =  data["data"]["abusive_info"]["充电桩输出电流异常比例"]["结果值"]
                    # 其他信息
                    dir_data['remaining_capacity'] =  float(data["data"]["soh_info"]["蓄电池最高单体电芯剩余容量"]["结果值"])
                    dir_data['soh'] =  float(data["data"]["soh_info"]["蓄电池容量SOH"]["结果值"])
                    if data["data"]["soh_info"]["蓄电池容量损失速率"]["结果值"] != '/':
                        dir_data['capacity_decay_rate'] =  float(data["data"]["soh_info"]["蓄电池容量损失速率"]["结果值"])
                    else:
                        dir_data['capacity_decay_rate'] = np.nan

                    dir_data['bms_stop_rsn'] =  data["data"]["alarm_management_info"]["BMS中止充电原因"]["结果值"]
                    
                    if data["data"]["alarm_management_info"]["充电机中止充电原因"]["结果值"] == "达到充电机设定的条件中止":
                        dir_data['charging_stop_rsn'] = 1
                    elif data["data"]["alarm_management_info"]["充电机中止充电原因"]["结果值"] == "人工中止":
                        dir_data['charging_stop_rsn'] = 2
                    elif data["data"]["alarm_management_info"]["充电机中止充电原因"]["结果值"] == "故障中止":
                        dir_data['charging_stop_rsn'] = 3
                    else:
                        dir_data['charging_stop_rsn'] = 4

                    # dir_data['charging_stop_rsn'] =  data["data"]["alarm_management_info"]["充电机中止充电原因"]["结果值"]

                    dir_data['charging_conv_efficiency'] =  float(data["data"]["charging_management_system_info"]["充电转换效率"]["结果值"])

                    dir_data['total_score'] =  float(data["data"]["score_info"]["总评分"]["结果值"])

                    data_list.append(dir_data)
        except Exception as e:
            print(f"Error: {e}")
            continue
    # 将列表转换为pandas DataFrame
    df = pd.DataFrame(data_list)

    # need_cov_float = ['max_cell_vol_dec', 'charging_time','high_rate_time', 'charging_capacity','charging_kwh']
    # df[need_cov_float] = df[need_cov_float].astype(float)
    # df need_cov_int 中指定的这些列的数据类型将被更改为整数
    # need_cov_int = []
    # df[need_cov_int] = df[need_cov_int].astype(int)
    return df


def plot_distribution(df, col_name, plot_kind, xlabel, ylabel, bins= None):
    # for col_name in df.columns.tolist():
    plt.figure(figsize=(10, 5))
    
    # if bins > 0:
    #     # need_plot_data.plot(kind=plot_kind, bins=bins)
    #     df[col_name].hist(bins=bins)
    # else:
    #     # need_plot_data.plot(kind=plot_kind)
    #     df[col_name].hist()
    if bins is not None:
        # 对数据进行分组
        grouped_data = pd.cut(df[col_name], bins=bins, right=False)
        # 计算每个区间的计数
        need_plot_data = grouped_data.value_counts().sort_index()
    else:
       need_plot_data = df[col_name].value_counts()
    
    need_plot_data.plot(kind= plot_kind,width=0.8)

    plt.title(col_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0, ha='right') 
    plt.savefig(f"/root/data/sp/charging_project_build/data/json/{xlabel}.png")

def plot_pie(df ,col_name, title):
    value_counts = df[col_name].value_counts(normalize=False)
    # 绘制成饼形图
    plt.pie(value_counts, labels=value_counts.index, autopct='%3.9f%%')
    plt.title(title)
    plt.savefig(f"/root/data/sp/charging_project_build/data/json/{col_name+'_pie'}.png")


def plot_distribution_bar_and_pie(df, col_name, xlabel, ylabel, bins= None):
    # for col_name in df.columns.tolist():
    plt.figure(figsize=(19, 6))
     
    # 左侧：柱状图
    plt.subplot(1, 2, 1)
    # 使用pd.cut将数据分组为区间
    if bins is not None:
        grouped_data = pd.cut(df[col_name], bins=bins)
    else:
        grouped_data = df[col_name]
    # 计算每个区间的数量
    interval_counts = grouped_data.value_counts().sort_index()
    # 绘制柱状图
    sns.barplot(x=interval_counts.index.astype(str), y=interval_counts.values, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{col_name} {ylabel} Distribution (Bar)')

    # 计算每个区间的数量
    if bins is not None:
        distribution = pd.cut(df[col_name], bins=bins).value_counts(sort=False).reindex(pd.IntervalIndex.from_breaks(bins))
    else:
        distribution = df[col_name].value_counts(sort=False)
    counts = distribution.fillna(0)  # 确保所有区间都有数据，即使数量为0
    # 过滤掉数量为0的区间
    non_zero_counts = counts[counts > 0]
    # 右侧：饼图``
    plt.subplot(1, 2, 2)
    labels = non_zero_counts.index.astype(str)  # 将区间转换为字符串标签
    sizes = non_zero_counts.values
    # plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.pie(sizes, autopct='%3.1f%%', pctdistance=1.2, startangle=200)
    plt.title(f'{col_name} Distribution (Pie)')
    
    plt.tight_layout()
    plt.savefig(f"/root/data/sp/charging_project_build/data/json/{xlabel}.png")


def plot_scatter(df, x_col_name, y_col_name, xlabel, ylabel, xcut, ycut):
    # 提取数据
    # 去掉 df 中包含 NaN 值的行
    df_cleaned = df.dropna(subset=[x_col_name, y_col_name])

    # 获取 x 和 y 的值
    x = df_cleaned[x_col_name]
    y = df_cleaned[y_col_name]
    
    x_data = np.array(x).reshape(-1, 1)  # 充电效率数据
    y_data = np.array(y)  # 电流（或其他变量）数据
    # 拟合线性回归模型
    linear_model = LinearRegression()
    linear_model.fit(x_data, y_data)
    # 拟合曲线
    y_pred = linear_model.predict(x_data)
    # 识别异常点（基于SOC分段，每隔5%循环检测）
    anomalies = np.zeros(len(x_data), dtype=bool)
    soc_bins = np.arange(0, 105, 5)  # 0-100 每隔5%
    prev_mask = None
    
    for i in range(len(soc_bins) - 1):
        mask = (x > soc_bins[i]) & (x <= soc_bins[i + 1])
        
        if np.sum(mask) < 2:
            if prev_mask is not None:
                mask = mask | prev_mask  # 尝试与上一个区间合并
            next_mask = (x >= soc_bins[i + 1]) & (x < soc_bins[i + 2]) if i + 2 < len(soc_bins) else None
            if next_mask is not None and np.sum(mask) < 2:
                mask = mask | next_mask  # 尝试与下一个区间合并
        
        if np.sum(mask) < 2:
            continue
        
        y_values = y[mask]
        
        # 进行聚类分析（2类）
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        y_values_reshaped = y_values.values.reshape(-1, 1)  # 如果 y_values 是 numpy 数组，直接 reshape
        kmeans.fit(y_values_reshaped)
        labels = kmeans.labels_

        # 计算两类的均值
        mean_0 = np.mean(y_values[labels == 0])
        mean_1 = np.mean(y_values[labels == 1])

        # 根据均值确定高值类和低值类
        if mean_0 > mean_1:
            high_label = 0
            low_label = 1
        else:
            high_label = 1
            low_label = 0

        # 计算均值更高类的最小值和均值更低类的最大值
        min_h = np.min(y_values[labels == high_label])
        max_l = np.max(y_values[labels == low_label])

        # # 动态计算异常阈值
        # threshold_diff = 0.1

        # # 判断是否满足异常条件
        # if abs(min_h - max_l) > threshold_diff:
        #     anomalies[mask] = labels == high_label

        # 额外条件：y值超过ycut也视为异常值
        anomalies[mask] = anomalies[mask] | (y[mask] > ycut)
        
        prev_mask = mask  # 记录上一个区间
    
    # 统计各区域点数
    total_points = len(df)
    q1 = ((x >= xcut) & (y >= ycut)).sum()
    q2 = ((x < xcut) & (y >= ycut)).sum()
    q3 = ((x < xcut) & (y < ycut)).sum()
    q4 = ((x >= xcut) & (y < ycut)).sum()
    
    # 计算占比
    q1_ratio = round((q1 / total_points) * 100, 2)
    q2_ratio = round((q2 / total_points) * 100, 2)
    q3_ratio = round((q3 / total_points) * 100, 2)
    q4_ratio = round((q4 / total_points) * 100, 2)
    
    # 计算异常值的数量
    anomaly_count = anomalies.sum()
    anomaly_count_percent = round(anomaly_count/total_points*100,2)
    
    # 画散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x[~anomalies], y[~anomalies], color='blue', alpha=0.5, label='Normal Points')
    plt.scatter(x[anomalies], y[anomalies], color='red', marker='x', label='Abnormal Points')
    plt.plot(x_data, y_pred, color='red', label='Linear Fit: y = {:.2f}x + {:.2f}'.format(linear_model.coef_[0], linear_model.intercept_))

    # 画分割线
    # plt.axvline(x=xcut, color='r', linestyle='--')
    # plt.axhline(y=ycut, color='r', linestyle='--')
    
    # 标注占比
    # plt.text(xcut + 0.02 * (x.max() - x.min()), ycut + max(0.02 * (y.max() - ycut), 0.01), f'{q1_ratio}%', fontsize=12, color='black')
    # plt.text(xcut - 0.2 * (x.max() - x.min()), ycut + max(0.02 * (y.max() - ycut), 0.01), f'{q2_ratio}%', fontsize=12, color='black')
    # plt.text(xcut - 0.2 * (x.max() - x.min()), ycut - max(0.02 * (ycut - y.min()), 0.08), f'{q3_ratio}%', fontsize=12, color='black')
    # plt.text(xcut + 0.02 * (x.max() - x.min()), ycut - max(0.02 * (ycut - y.min()), 0.08), f'{q4_ratio}%', fontsize=12, color='black')
    
    # 设置标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Proportion of abnormal points: {anomaly_count_percent}%')
    plt.legend()
    
    # 保存图片
    save_path = f"/root/data/sp/charging_project_build/data/json/{ylabel}_{xlabel}_scatter.png"
    plt.savefig(save_path)
    plt.close() 
    
    print(f"Scatter plot saved at: {save_path}")



def plot_scatter_res(df, x_col_name, y_col_name, xlabel, ylabel, xcut, ycut):
    # 提取数据
    # 去掉 df 中包含 NaN 值的行
    df_cleaned = df.dropna(subset=[x_col_name, y_col_name])

    # 获取 x 和 y 的值
    x = df_cleaned[x_col_name]
    y = df_cleaned[y_col_name]
    
    x_data = np.array(x).reshape(-1, 1)  # 充电效率数据
    y_data = np.array(y)  # 电流（或其他变量）数据
    
    # 识别异常点（基于SOC分段，每隔5%循环检测）
    anomalies = np.zeros(len(x_data), dtype=bool)
    soc_bins = np.arange(0, 105, 5)  # 0-100 每隔5%
    prev_mask = None
    
    for i in range(len(soc_bins) - 1):
        mask = (x > soc_bins[i]) & (x <= soc_bins[i + 1])
        
        if np.sum(mask) < 2:
            if prev_mask is not None:
                mask = mask | prev_mask  # 尝试与上一个区间合并
            next_mask = (x >= soc_bins[i + 1]) & (x < soc_bins[i + 2]) if i + 2 < len(soc_bins) else None
            if next_mask is not None and np.sum(mask) < 2:
                mask = mask | next_mask  # 尝试与下一个区间合并
        
        if np.sum(mask) < 2:
            continue
        # 额外条件：y值超过ycut也视为异常值
        anomalies[mask] = anomalies[mask] | (y[mask] > ycut)
        
        prev_mask = mask  # 记录上一个区间
    
    # 统计各区域点数
    total_points = len(df)
    q1 = ((x >= xcut) & (y >= ycut)).sum()
    q2 = ((x < xcut) & (y >= ycut)).sum()
    q3 = ((x < xcut) & (y < ycut)).sum()
    q4 = ((x >= xcut) & (y < ycut)).sum()
    
    # 计算异常值的数量
    anomaly_count = anomalies.sum()
    anomaly_count_percent = round(anomaly_count/total_points*100,2)
    
    # 画散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x[~anomalies], y[~anomalies], color='blue', alpha=0.5, label='Normal Points')
    plt.scatter(x[anomalies], y[anomalies], color='red', marker='x', label='Abnormal Points')

    # 画分割线
    # plt.axvline(x=xcut, color='r', linestyle='--')
    # plt.axhline(y=ycut, color='r', linestyle='--')
    
    # 标注占比
    # plt.text(xcut + 0.02 * (x.max() - x.min()), ycut + max(0.02 * (y.max() - ycut), 0.01), f'{q1_ratio}%', fontsize=12, color='black')
    # plt.text(xcut - 0.2 * (x.max() - x.min()), ycut + max(0.02 * (y.max() - ycut), 0.01), f'{q2_ratio}%', fontsize=12, color='black')
    # plt.text(xcut - 0.2 * (x.max() - x.min()), ycut - max(0.02 * (ycut - y.min()), 0.08), f'{q3_ratio}%', fontsize=12, color='black')
    # plt.text(xcut + 0.02 * (x.max() - x.min()), ycut - max(0.02 * (ycut - y.min()), 0.08), f'{q4_ratio}%', fontsize=12, color='black')
    
    # 设置标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Proportion of abnormal points: {anomaly_count_percent}%')
    plt.legend()
    
    # 保存图片
    save_path = f"/root/data/sp/charging_project_build/data/json/{ylabel}_{xlabel}_scatter.png"
    plt.savefig(save_path)
    plt.close() 
    
    print(f"Scatter plot saved at: {save_path}")
# point_location 为列表，表示x，y cut线偏移度，[x_right, x_left, y_up, y_down]
def plot_scatter_double_y(df, x_col_name, y_col_name, y_col_name2, xlabel, ylabel, ylabel2, xcut, ycut, y_cut2, point_location):
    # 提取数据
    # 去掉 df 中包含 NaN 值的行
    df_cleaned = df.dropna(subset=[x_col_name, y_col_name])

    # 获取 x 和 y 的值
    x = df_cleaned[x_col_name]
    y1 = df_cleaned[y_col_name]
    y2 = df_cleaned[y_col_name2]
    
    # 识别异常点（基于SOC分段，每隔5%循环检测）
    anomalies = np.zeros(len(x), dtype=bool)
    anomalies2 = np.zeros(len(x), dtype=bool)

    soc_bins = np.arange(0, 105, 5)  # 0-100 每隔5%
    prev_mask = None
    
    for i in range(len(soc_bins) - 1):
        mask = (x > soc_bins[i]) & (x <= soc_bins[i + 1])
        
        if np.sum(mask) < 2:
            if prev_mask is not None:
                mask = mask | prev_mask  # 尝试与上一个区间合并
            next_mask = (x >= soc_bins[i + 1]) & (x < soc_bins[i + 2]) if i + 2 < len(soc_bins) else None
            if next_mask is not None and np.sum(mask) < 2:
                mask = mask | next_mask  # 尝试与下一个区间合并
        
        if np.sum(mask) < 2:
            continue
        
        # y_values = y1[mask]
        
        # # 进行聚类分析（2类）
        # kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        # y_values_reshaped = y_values.values.reshape(-1, 1)  # 如果 y_values 是 numpy 数组，直接 reshape
        # kmeans.fit(y_values_reshaped)
        # labels = kmeans.labels_

        # # 计算两类的均值
        # mean_0 = np.mean(y_values[labels == 0])
        # mean_1 = np.mean(y_values[labels == 1])

        # # 根据均值确定高值类和低值类
        # if mean_0 > mean_1:
        #     high_label = 0
        #     low_label = 1
        # else:
        #     high_label = 1
        #     low_label = 0

        # 计算均值更高类的最小值和均值更低类的最大值
        # min_h = np.min(y_values[labels == high_label])
        # max_l = np.max(y_values[labels == low_label])

        # # 动态计算异常阈值
        # threshold_diff = 0.1

        # # 判断是否满足异常条件
        # if abs(min_h - max_l) > threshold_diff:
        #     anomalies[mask] = labels == high_label

        # 额外条件：y值超过ycut也视为异常值
        anomalies[mask] = anomalies[mask] | (y1[mask] > ycut)
        anomalies2[mask] = anomalies2[mask] | (y2[mask] > y_cut2)
        prev_mask = mask  # 记录上一个区间
    
    # 统计各区域点数
    total_points = len(df)
    q1 = ((x >= xcut) & (y1 >= ycut)).sum()
    q2 = ((x < xcut) & (y1 >= ycut)).sum()
    q3 = ((x < xcut) & (y1 < ycut)).sum()
    q4 = ((x >= xcut) & (y1 < ycut)).sum()
    
    # 计算占比
    q1_ratio = round((q1 / total_points) * 100, 2)
    q2_ratio = round((q2 / total_points) * 100, 2)
    q3_ratio = round((q3 / total_points) * 100, 2)
    q4_ratio = round((q4 / total_points) * 100, 2)

    # y2轴的占比
    q1 = ((x >= xcut) & (y2 >= y_cut2)).sum()
    q2 = ((x < xcut) & (y2 >= y_cut2)).sum()
    q3 = ((x < xcut) & (y2 < y_cut2)).sum()
    q4 = ((x >= xcut) & (y2 < y_cut2)).sum()
    
    # 计算占比
    q1_ratio2 = round((q1 / total_points) * 100, 2)
    q2_ratio2 = round((q2 / total_points) * 100, 2)
    q3_ratio2 = round((q3 / total_points) * 100, 2)
    q4_ratio2 = round((q4 / total_points) * 100, 2)
    
    # 计算异常值的数量
    anomaly_count = anomalies.sum()
    anomaly_count_percent = round(anomaly_count/total_points*100,2)
    
    # 画散点图
    # plt.figure(figsize=(8, 6))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.scatter(x[~anomalies], y1[~anomalies], color='blue', alpha=0.5, label='Normal Points')
    ax1.scatter(x[anomalies], y1[anomalies], color='red', marker='x', label='Abnormal Points')
    
    # 画分割线
    ax1.axvline(x=xcut, color='r', linestyle='--')
    ax1.axhline(y=ycut, color='r', linestyle='--')
    
    # 标注占比
    # ax1.text(xcut + point_location[0] * (x.max() - x.min()), ycut + max(point_location[2] * (y1.max() - ycut), point_location[3]), f'{q1_ratio}%', fontsize=12, color='black')
    # ax1.text(xcut - point_location[1] * (x.max() - x.min()), ycut + max(point_location[2] * (y1.max() - ycut), point_location[3]), f'{q2_ratio}%', fontsize=12, color='black')
    # ax1.text(xcut - point_location[1] * (x.max() - x.min()), ycut - max(point_location[2] * (ycut - y1.min()), point_location[4]), f'{q3_ratio}%', fontsize=12, color='black')
    # ax1.text(xcut + point_location[0] * (x.max() - x.min()), ycut - max(point_location[2] * (ycut - y1.min()), point_location[4]), f'{q4_ratio}%', fontsize=12, color='black')
    
    # 创建第二个 y 轴
    ax2 = ax1.twinx()  # 共享 x 轴
    ax2.axhline(y=y_cut2, color='purple', linestyle='--')
    # 绘制 y2 的散点图
    ax2.set_ylabel(ylabel2)
    ax2.scatter(x[~anomalies2], y2[~anomalies2], color='orange', alpha=0.5, label='Normal Points')
    ax2.scatter(x[anomalies2], y2[anomalies2], color='purple', marker='x', label='Abnormal Points')
                                 
    # 标注占比
    # ax2.text(xcut + point_location[0] * (x.max() - x.min()), y_cut2 + max(point_location[5] * (y2.max() - y_cut2), point_location[6]), f'{q1_ratio2}%', fontsize=12, color='green')
    # ax2.text(xcut - point_location[1] * (x.max() - x.min()), y_cut2 + max(point_location[5] * (y2.max() - y_cut2), point_location[6]), f'{q2_ratio2}%', fontsize=12, color='green')
    # ax2.text(xcut - point_location[1] * (x.max() - x.min()), y_cut2 - max(point_location[5] * (y_cut2 - y2.min()), point_location[7]), f'{q3_ratio2}%', fontsize=12, color='green')
    # ax2.text(xcut + point_location[0] * (x.max() - x.min()), y_cut2 - max(point_location[5] * (y_cut2 - y2.min()), point_location[7]), f'{q4_ratio2}%', fontsize=12, color='green')
    
    # 设置标签
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.title(f'Proportion of abnormal points')
    #plt.legend()
    # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.legend(loc="upper left", bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes)
    fig.tight_layout()  # 防止右上角的标签被剪裁
    # 保存图片
    save_path = f"/root/data/sp/charging_project_build/data/json/{ylabel2}_{xlabel}_scatter.png"
    plt.savefig(save_path)
    plt.close()
    

def  plot_brand_compare_scatter(df, col_name, title, xlabel, y_lable, plot_name, y_cut= None):
   # 设置图形大小
    plt.figure(figsize=(16, 9))
    # 使用 seaborn 绘制散点图，按 brand 分组并设置不同颜色
    # 创建一个包含30种颜色的调色板
    unique_brands = df['brand'].unique()
    palette = sns.color_palette("husl", len(unique_brands))  # 调整亮度范围以增强对比度
   
#   线型
#     style_order = [
#     '+',    # 加号
#     'x',    # 叉号
#     '1',    # 下三角
#     '2',    # 上三角
#     '3',    # 左三角
#     '4'    # 右三角
# ]

#  
    style_order = [
    'o',    
    's',    
    '^',    
    'v',   
    '*',    
    'p'    
]
    # 确保有足够多的标记
    markers = style_order * (len(unique_brands) // len(style_order) + 1)
    markers = markers[:len(unique_brands)]

    
    sns.scatterplot(x='soc_end', y= col_name, hue='brand', style='brand', data=df,
                    palette=palette,markers=markers, s=100)
    # 设置横轴范围 0-100，以 5% 为间隔
    soc_min = df['soc_end'].min()
    plt.xticks(np.arange(soc_min, 100, 5))
    
    # 设置纵轴范围为 df['vol_dec'] 的最小值和最大值
    plt.ylim(df[col_name].min(), df[col_name].max()+0.1)

    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(y_lable, fontsize=14)
    if y_cut is not None:
        plt.axhline(y=y_cut, color='black', linestyle='--')
    # 显示图例
    plt.legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
   
    # 自动调整布局
    plt.tight_layout()
    # 保存图片
    save_path = f"/root/data/sp/charging_project_build/data/json/{plot_name}_scatter.png"
    plt.savefig(save_path)
    plt.close()


def  plot_brand_compare_scatter_ef(df, col_name, title, xlabel, y_lable, plot_name, y_cut= None):
   # 设置图形大小
    plt.figure(figsize=(8, 4))
    # 使用 seaborn 绘制散点图，按 brand 分组并设置不同颜色
    # 创建一个包含30种颜色的调色板
    unique_brands = df['brand'].unique()
    palette = sns.color_palette("husl", len(unique_brands))  # 调整亮度范围以增强对比度
   
#   线型
#     style_order = [
#     '+',    # 加号
#     'x',    # 叉号
#     '1',    # 下三角
#     '2',    # 上三角
#     '3',    # 左三角
#     '4'    # 右三角
# ]

#  
    style_order = [
    'o',    
    's',    
    '^',    
    'v',   
    '*',    
    'p'    
]
    
    # 确保有足够多的标记
    markers = style_order * (len(unique_brands) // len(style_order) + 1)
    markers = markers[:len(unique_brands)]


    sns.scatterplot(x='charging_conv_efficiency', y= col_name, hue='brand', style='brand', data=df,
                    palette=palette,markers=markers, s=100)
    
    
    # 新增曲线拟合
    # x_data = np.array(df['charging_conv_efficiency']).reshape(-1, 1)  # 充电效率数据
    # y_data = np.array(df[col_name])  # 电流（或其他变量）数据
    # # 拟合线性回归模型
    # linear_model = LinearRegression()
    # linear_model.fit(x_data, y_data)
    # # 拟合曲线
    # y_pred = linear_model.predict(x_data)
    # #plt.plot(x_data, y_pred, color='red', label='Linear Fit: y = {:.2f}x + {:.2f}'.format(linear_model.coef_[0], linear_model.intercept_))
    # plt.plot(x_data, y_pred, color='red', label='Linear Fit')
    # 设置横轴范围 0-100，以 5% 为间隔
    soc_min = df['charging_conv_efficiency'].min()
    plt.xticks(np.arange(soc_min, 100, 5))
    
    # 设置纵轴范围为 df['vol_dec'] 的最小值和最大值
    plt.ylim(df[col_name].min(), df[col_name].max()+0.1)

    # 添加标题和标签
    plt.title(title, fontsize=14, loc='left')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(y_lable, fontsize=14)
    if y_cut is not None:
        plt.axhline(y=y_cut, color='black', linestyle='--')
    # 显示图例
    plt.legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
   
    # 自动调整布局
    plt.tight_layout()
    # 保存图片
    save_path = f"/root/data/sp/charging_project_build/data/json/{plot_name}_scatter.png"
    plt.savefig(save_path)
    plt.close()



def plot_brand_compare_sn(df, sn_list, title, xlabel, y_lable, plot_name):
   # 设置图形大小
    plt.figure(figsize=(10, 6))
    # 使用 seaborn 绘制散点图，按 brand 分组并设置不同颜色
    
    # 创建一个颜色数组，用于区分不同的充电桩
    colors = plt.cm.get_cmap('Set1', len(sn_list))
    # colors = ['red', 'green', 'orange', 'blue']
    # 循环遍历每个充电桩号
    for i, sn in enumerate(sn_list):
        # 筛选出该充电桩号的所有数据
        df_sn = df[df['sn'] == sn]
        current_data = df_sn["total_cur_dec_mid"]
        # 对应的充电效率数据点
        efficiency_data = df_sn['charging_conv_efficiency']
        
        # 绘制散点图
        # plt.scatter(current_data, efficiency_data, color=colors(i), label=f'SN: {sn}')
        plt.scatter(efficiency_data , current_data, color=colors(i), label=f'SN: {sn}')
    
        # 新增曲线拟合
        x_data = np.array(efficiency_data).reshape(-1, 1)  # 充电效率数据
        y_data = np.array(current_data)  # 电流（或其他变量）数据
        # 拟合线性回归模型
        linear_model = LinearRegression()
        linear_model.fit(x_data, y_data)
        # 拟合曲线
        y_pred = linear_model.predict(x_data)
        # plt.plot(x_data, y_pred, color=colors(i), label='Linear Fit: y = {:.2f}x + {:.2f}'.format(linear_model.coef_[0], linear_model.intercept_))
        plt.plot(x_data, y_pred, color=colors(i))
    # 设置横轴范围 0-100，以 5% 为间隔
    
    # 设置标题和坐标轴标签
    # plt.title('charging_conv_efficiency vs cur_absolute_error')
    # plt.xlabel('cur (A)')
    # plt.ylabel('charging_conv_efficiency (%)')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(y_lable)
    # 显示图例
    plt.legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
   
    # 自动调整布局
    plt.tight_layout()
    # 保存图片
    save_path = f"/root/data/sp/charging_project_build/data/json/{plot_name}_scatter.png"
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    # # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False 

    df_json = get_data_from_json(json_dir)
    df_json['VIN'] = df_json['VIN'].replace('/', np.nan)

    # df = df_json.dropna(subset=['VIN'])    # 去除缺失值
    # df = df.drop_duplicates(subset=['VIN'])  # 去除重复值
    # # df['vin'].to_csv('vin.txt', index=False, header=False)
    df_csv_vin = get_data_from_csv(csv_dir_path)
    # filtered_json_df = pd.merge(df_json, df_csv_vin[['VIN']], on='VIN', how='inner')
    filtered_df = pd.merge(df_json, df_csv_vin, on='VIN', how='inner')  #合并df_json和df_csv_vin, 取VIN相同的行交集
    filtered_df = filtered_df.drop_duplicates(subset=['file_name'])  
    du_vin = filtered_df.groupby('VIN').filter(lambda x: len(x) > 1)

    # 去掉数据上传不对的temp值
    filtered_df_for_temp = filtered_df[(filtered_df['htemp_start_change_need_seconds'] < 500) & (filtered_df['htemp_start_change_need_seconds'] > 0)]
    plot_scatter_x_y(filtered_df_for_temp['htemp_start_change_need_seconds'], filtered_df_for_temp['htemp_start_cur'], "change_time v.s. cur", "first_change_time", "cur", "first_change_time_cur","/root/data/sp/charging_project_build/data/json/")
    # 看一下直方图的 温度变化时间分布
    plot_distribution_bar_and_pie(filtered_df_for_temp, "htemp_start_change_need_seconds", "htemp_start_change_need_seconds", "count", [0,30,50,100,150,200,250,300,350,400,450,500])
    plot_distribution_bar_and_pie(filtered_df_for_temp, "htemp_start_cur", "htemp_start_cur", "count", [0,30,50,100,150,200,250,300,350,400,450,500])
    # 看一下线阻分布的直方图 
    plot_distribution_bar_and_pie(filtered_df, "line_res", "line_res", "count", [0,0.01,0.02,0.03,0.04,0.05,0.08,0.10,0.15,0.20])
     
    # 获取所有的充电桩号
    sn_list = filtered_df['sn'].unique()
    sn_list_1 = [sn_list[8], sn_list[1], sn_list[5], sn_list[2]]
    # 循环遍历每个充电桩号
    for sn in sn_list:
        # 筛选出该充电桩号的所有数据
        df_sn = filtered_df[filtered_df['sn'] == sn]
        # 充电效率-线阻分布图
        plot_scatter_res(df_sn, "charging_conv_efficiency", "line_res", "charging_conv_efficiency(%)", f"{sn}_line_res(V)", 85,0.05)
        # 充电效率-电压分布图
        # plot_scatter(df_sn, "charging_conv_efficiency", "total_vol_dec_mid", "charging_conv_efficiency(%)", f"{sn}_tvol_dec_mid(V)", 85,5)
        # 充电效率-电流分布图
        # plot_scatter(df_sn, "charging_conv_efficiency", "total_cur_dec_mid", "charging_conv_efficiency(%)", f"{sn}_tcur_dec_mid(A)", 85, 15)
        
        # 品牌分析：充电效率-电流分布图
        # plot_brand_compare_scatter_ef(df_sn, 'total_cur_dec_mid', 'charging_conv_efficiency v.s. cur_absolute_error', 'charging_conv_efficiency(%)', 'cur_absolute_error(A)',f"{sn}_tcur_absolute_error(A)", 15)
        # plot_brand_compare_scatter_ef(df_sn, 'total_vol_dec_mid', 'charging_conv_efficiency v.s. vol_absolute_error', 'charging_conv_efficiency(%)', 'vol_absolute_error(V)',f"{sn}_tvol_absolute_error(V)", 5)

    plot_brand_compare_sn(filtered_df, sn_list_1, 'charging_conv_efficiency v.s. cur_absolute_error', 'charging_conv_efficiency(%)', 'cur_absolute_error(A)', 'sn_tcur_absolute_error(A)')

# 各类型分布自定义区间
    custom_bins_vol_dec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0]
    custom_bins_soc = [0, 20, 40, 60, 80, 100]
    custom_bins_temp = [-20, 0, 20, 40, 60, 80]
    custom_bins_decay_rate = [0, 5, 10, 15, 20, 25]
    custom_bins_efficiency = [0, 70, 75, 80, 85, 90, 95, 100]
    custom_bins_score = [50,60,70,80,90,100]
    custom_bins_charing_stop_rsn = ["达到充电机设定的条件中止", "故障中止", "人工中止", "无"]
    
    
    
    # # 统计特斯拉车型
    tesla_df = filtered_df[filtered_df['brand'] == "特斯拉"]
    # 充电效率-线阻分布图
    plot_scatter_res(tesla_df, "charging_conv_efficiency", "line_res", "charging_conv_efficiency(%)", "tesla_line_res", 85,0.05)
    # # 充电效率-fuel_efficiency: BMS充电电压, 电流 与 充电桩充电电压电流之间的关系
    # plot_scatter(tesla_df, "charging_conv_efficiency", "total_vol_dec_mid", "charging_conv_efficiency(%)", "tesla_total_vol_dec_mid(V)", 85,5)
    # # plot_scatter_double_y(tesla_df, "charging_conv_efficiency", "total_vol_dec_mid", "total_cur_dec_mid", "charging_conv_efficiency(%)", "tesla_total_vol_dec_mid(V)", "tesla_total_cur_dec_mid(A)", 90, 3, 15,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    # # soc-电压极差, 温度极差分布
    # plot_scatter_double_y(tesla_df, "soc_end", "max_cell_vol_dec", "max_temp_dec", "soc(%)", "tesla_vol_dec(V)", "tesla_temp_dec(°C)", 80, 0.5, 15,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    # # 充电停止原因分布
    # plot_scatter_double_y(tesla_df, "soc_end", "max_cell_vol_max", "charging_stop_rsn", "soc(%)", "tesla_max_cell_vol(V)", "tesla_charging_stop_rsn", 80, 4.30, 2,[0.02,0.2,0.02,0.01,0.08,0.1,0.1,0.5])
    # # 报告评分与 电压极差，容量的关系
    # #plot_scatter_double_y(tesla_df, "total_score", "max_cell_vol_dec", "soh", "total_score", "tesla_max_cell_vol_dec(V)", "soh(%)", 85, 0.3, 80,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    
    # # 统计比亚迪车型的电压极差值分布
    # byd_df = filtered_df[filtered_df['brand'] == "比亚迪"]
    # # 充电效率-fuel_efficiency: BMS充电电压, 电流 与 充电桩充电电压电流之间的关系
    # plot_scatter_double_y(byd_df, "charging_conv_efficiency", "total_vol_dec_mid", "total_cur_dec_mid", "charging_conv_efficiency(%)", "byd_total_vol_dec_mid(V)", "byd_total_cur_dec_mid(A)", 90, 5, 15,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    # # soc-电压极差, 温度极差分布
    # plot_scatter_double_y(byd_df, "soc_end", "max_cell_vol_dec", "max_temp_dec", "soc(%)", "byd_vol_dec(V)", "byd_temp_dec(°C)", 80, 0.5, 15,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    # # 充电停止原因分布
    # plot_scatter_double_y(byd_df, "soc_end", "max_cell_vol_max", "charging_stop_rsn", "soc(%)", "byd_max_cell_vol(V)", "byd_charging_stop_rsn", 80, 4.30, 2,[0.02,0.2,0.02,0.01,0.08,0.1,0.1,0.5])
    # # 报告评分与 电压极差，容量的关系
    
    # # 统计小鹏车型的电压极差值分布
    xp_df = filtered_df[filtered_df['brand'] == "小鹏"]
    # 充电效率-线阻分布图
    plot_scatter_res(xp_df, "charging_conv_efficiency", "line_res", "charging_conv_efficiency(%)", "xp_line_res", 85,0.05)
    # # 充电效率-fuel_efficiency: BMS充电电压, 电流 与 充电桩充电电压电流之间的关系
    # plot_scatter_double_y(xp_df, "charging_conv_efficiency", "total_vol_dec_mid", "total_cur_dec_mid", "charging_conv_efficiency(%)", "xp_total_vol_dec_mid(V)", "xp_total_cur_dec_mid(A)", 90, 3, 15,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    # # soc-电压极差, 温度极差分布
    # plot_scatter_double_y(xp_df, "soc_end", "max_cell_vol_dec", "max_temp_dec", "soc(%)", "xp_vol_dec(V)", "xp_temp_dec(°C)", 80, 0.5, 15,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    # # 充电停止原因分布
    # plot_scatter_double_y(xp_df, "soc_end", "max_cell_vol_max", "charging_stop_rsn", "soc(%)", "xp_max_cell_vol(V)", "xp_charging_stop_rsn", 80, 4.30, 2,[0.02,0.2,0.02,0.01,0.08,0.1,0.1,0.5])
    # # 报告评分与 电压极差，容量的关系

    # # 其他车型的分布
    # # 筛选出不包含特斯拉、比亚迪和小鹏的车型
    # brands_to_exclude = ["特斯拉", "比亚迪", "小鹏"]
    # filtered_df_excluded = filtered_df[~filtered_df['brand'].isin(brands_to_exclude)] 
    # # 充电效率-fuel_efficiency: BMS充电电压, 电流 与 充电桩充电电压电流之间的关系
    # plot_scatter_double_y(filtered_df_excluded, "charging_conv_efficiency", "total_vol_dec_mid", "total_cur_dec_mid", "charging_conv_efficiency(%)", "other_total_vol_dec_mid(V)", "other_total_cur_dec_mid(A)", 90, 3, 15,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    # # soc-电压极差, 温度极差分布
    # plot_scatter_double_y(filtered_df_excluded, "soc_end", "max_cell_vol_dec", "max_temp_dec", "soc(%)", "other_vol_dec(V)", "other_temp_dec(°C)", 80, 0.5, 15,[0.02,0.2,0.02,0.01,0.08,0.1,0.5,0.5])
    # # 充电停止原因分布
    # plot_scatter_double_y(filtered_df_excluded, "soc_end", "max_cell_vol_max", "charging_stop_rsn", "soc(%)", "other_max_cell_vol(V)", "other_charging_stop_rsn", 80, 4.30, 2,[0.02,0.2,0.02,0.01,0.08,0.1,0.1,0.5])
    # # 报告评分与 电压极差，容量的关系
    

    # 不同车品牌的分析
    brands_to_exclude = ["特斯拉"]
    filtered_df_excluded = filtered_df[~filtered_df['brand'].isin(brands_to_exclude)] 
    plot_brand_compare_scatter(filtered_df_excluded, 'max_cell_vol_dec', 'SOC vs Vol Dec', 'SOC(%)', 'Vol Dec(V)','brand_vol_dec', 0.5)
    #plot_brand_compare_scatter(filtered_df_excluded, 'max_temp_dec', 'SOC vs Temp Dec (Grouped by Brand)', 'SOC(%)', 'Temp Dec(°C)','brand_temp_dec', 15)
    #plot_brand_compare_scatter(filtered_df, 'soh', 'SOC vs SOH (Grouped by Brand)', 'SOC(%)', 'SOH(%)','brand_soh', 80)
    plot_brand_compare_scatter(filtered_df, 'charging_conv_efficiency', 'SOC vs Charging Conv Efficiency', 'SOC(%)', 'Charging Conv Efficiency(%)','brand_charging_conv_efficiency', 90)
    print("统计分析结束")
    
    
         
