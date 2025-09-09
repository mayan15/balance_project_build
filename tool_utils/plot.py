# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:32:22 2025

@author: SP
"""
import traceback
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np

def set_plot_properties(border_padding=0.1, figsize=(8, 6), bwith=2, edge_type=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 设置图像的大小和边框间距
            fig, ax = plt.subplots(figsize=figsize)
            plt.subplots_adjust(left=border_padding, right=1-border_padding, top=1-border_padding, bottom=border_padding)
            if edge_type == None:
                edge = plt.gca()
            else:
                edge = edge_type

            edge.spines['left'].set_color('black') 
            edge.spines['bottom'].set_color('black')  
            edge.spines['right'].set_color('black')  
            edge.spines['top'].set_color('black')
            edge.spines['bottom'].set_linewidth(bwith)
            edge.spines['left'].set_linewidth(bwith)
            edge.spines['top'].set_linewidth(bwith)
            edge.spines['right'].set_linewidth(bwith)
            
            func(ax, *args, **kwargs)  # 调用原始绘图函数
            fig.tight_layout()         # 调整布局防止标签重叠
            plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.25,hspace=0.3, wspace=None) 
            plt.close() 
        return wrapper
    return decorator


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
        print(f"plot_data error: {traceback.print_exc()}")



@set_plot_properties(border_padding=0.15, figsize=(19, 6))
def plot_vol_current(ax, df, ax_fontsize, picture_name, picture_save_path):
    '''
    绘制电池组数据图
    参数：
    ax: 绘图对象
    df: 数据集
    '''
    try:
        indices = np.linspace(0, len(df['date']) - 1, 10, dtype=int)
        indices_time = df['date'].to_numpy()    #.iloc[indices].values
        time_x = df['date'].iloc[indices].values

        ax.plot(indices_time, df['vol_max'].to_numpy(), label='max_vol', color='red')   # 绘制第一个图，主纵坐标
        ax.plot(indices_time, df['vol_mid'].to_numpy(), label='mid_vol', color='green')   
        
        ax.set_ylim(min(df['vol_max'])*0.9, max(df['vol_max'])*1.1)
        ax.tick_params(axis='y', labelsize=ax_fontsize)  # 设置y 轴 标签字体大小
        ax.set_xticks(time_x)               
        ax.set_xticklabels(time_x)
        ax.set_xticklabels(ax.get_xticks(), fontsize= ax_fontsize)
        ax.set_ylabel('电压(V)', fontsize= ax_fontsize)
        ax.legend( loc='upper right', fontsize=ax_fontsize, bbox_to_anchor=(1.2, 0.7), frameon=False)  
        
        # 创建次要坐标轴
        ax2 = ax.twinx()  # 创建共享 x 轴的次轴
        ax2.plot(indices_time, df['current'].to_numpy(), label='current', linestyle='--', color='orangered')  # 仅为第一条线设置 label
        ax2.set_ylim(min(df['current'])*0.9,max(df['current'])*1.1)
        ax2.tick_params(axis='y', labelsize=ax_fontsize)  # 设置y 轴 标签字体大小
        ax2.set_ylabel('电流(A)', fontsize=ax_fontsize)  # 次纵坐标标签
        ax2.legend( loc='upper right', fontsize=ax_fontsize, bbox_to_anchor=(1.2, 0.5), frameon=False)  
    
        # 设置横坐标（x）时间显示的点
        ax2.set_xticks(time_x)               
        ax2.set_xticklabels(time_x)
        # ax2.set_xticklabels(ax2.get_xticks(), fontsize=ax_fontsize)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 显示 mm-dd hh 格式

        # 设置横坐标的主要刻度
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
        plt.text(0.5, 1.0, picture_name, fontsize=24, color='#23272C', \
                transform= ax.transAxes, verticalalignment='bottom', horizontalalignment='center') 
        plt.savefig(f"{picture_save_path}/{picture_name}.png")
          
    except Exception as e:
        print(f"plot_data error: {traceback.print_exc()}")


@set_plot_properties(border_padding=0.15, figsize=(19, 6))
def  plot_vol_number(ax, df, ax_fontsize, picture_name, picture_save_path):
    try:    
        indices = np.linspace(0, len(df['date']) - 1, 10, dtype=int)
        indices_time = df['date'].to_numpy()    #.iloc[indices].values
        time_x = df['date'].iloc[indices].values
        '''' # 绘制 电压-电池节号分布图 '''  
        ax.plot(indices_time, df['vol_max'].to_numpy(), label='max_vol', color='red')   # 绘制第一个图，主纵坐标
        ax.set_ylim(min(df['vol_max'])*0.9, max(df['vol_max'])*1.1)
        ax.tick_params(axis='y', labelsize=ax_fontsize)  # 设置y 轴 标签字体大小
        ax.set_xticks(time_x)               
        ax.set_xticklabels(time_x)
        ax.set_xticklabels(ax.get_xticks(), fontsize= ax_fontsize)
        ax.set_ylabel('电压(V)', fontsize= ax_fontsize)
        ax.legend( loc='upper right', fontsize=ax_fontsize, bbox_to_anchor=(1.2, 0.7),frameon=False)  # 在右侧显示图例

        ax2 = ax.twinx()  # 创建共享 x 轴的第二个 y 轴
        ax2.scatter(indices_time, df['max_vol_no'], color='blue', label='cell_no', zorder=5)

        ax2.tick_params(axis='y', labelsize=ax_fontsize)  # 设置y 轴 标签字体大小
        ax2.set_ylabel('电池节号', fontsize=ax_fontsize)  # 次纵坐标标签
        ax2.legend( loc='upper right', fontsize=ax_fontsize,bbox_to_anchor=(1.2, 0.5),frameon=False)  
    
        # 设置横坐标（x）时间显示的点
        ax2.set_xticks(time_x)               
        ax2.set_xticklabels(time_x)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 显示 mm-dd hh 格式

        # 设置横坐标的主要刻度
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
        plt.text(0.5, 1.0, picture_name, fontsize=24, color='#23272C', \
                transform= ax.transAxes, verticalalignment='bottom', horizontalalignment='center') 
        plt.savefig(f"{picture_save_path}/{picture_name}.png")

    except Exception as e:
        print(f"plot_data error: {traceback.print_exc()}")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

@set_plot_properties(border_padding=0.15)
def plot_scatter_x_y(ax, x_data, y_data, title, x_lable, y_lable, picture_name, picture_save_path):
    '''
    绘制普通的X,Y1
    参数：
    ax: 绘图对象
    x_data: x轴数据
    y_data: y轴数据
    ax_fontsize: 字体大小
    picture_name: 图片名称
    picture_save_path: 图片保存路径
    '''
    try:
        data = pd.DataFrame({
            'first_change_time': x_data,
            'cur': y_data
        })
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        dbscan = DBSCAN(eps=0.5, min_samples=5)  # 调整参数
        dbscan_labels = dbscan.fit_predict(scaled_data)

        # 将聚类标签添加到数据中
        data['cluster'] = dbscan_labels

        #plt.scatter(x_data, y_data, s=10, alpha=0.5, color='blue')
        plt.scatter(data['first_change_time'], data['cur'], c=data['cluster'], cmap='viridis')
        plt.colorbar(label='Cluster')
        plt.title(title)
        plt.xlabel(x_lable)
        plt.ylabel(y_lable)
        plt.savefig(f"{picture_save_path}/{picture_name}.png")
          
    except Exception as e:
        print(f"plot_data error: {traceback.print_exc()}")







