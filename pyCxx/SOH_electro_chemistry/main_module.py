import numpy as np
import pandas as pd

import time
import os
import traceback
import itertools  

"""
    log.logger.debug("用于打印日志")
"""
from tool_utils.pvlog import Logger, level_SOH_electro
log = Logger('./logs/SOH_electro.log', level=level_SOH_electro)

import warnings  
with warnings.catch_warnings(record=True):  
    warnings.simplefilter("ignore")    

"""
    storage_database : 所需数据类
"""
from .storage_database import Str_Info_for_Procure_Param, SOH_Output, Data_Used_to_Fit_Volt

"""
    cuda_main_param_LFP : 电化学参数辨识 - LFP
"""
from .call_electrochem_identify import param_identify_then_filter

"""
    SOH_distribution : SOH 分布 计算最终 SOH 值
"""
from .screen_param_by_soh_distribution import run_soh_distribution

"""
    sift_param_via_distance : 归一化 处理参数后 计算距离，选择参数
"""
from .sift_param_via_distance import params_range_search

"""
    ods_search_params : 全方向 多步长 搜索参数， 然后基于搜索的参数计算 该电池的 SOH
"""
from .ods_search_params import ods_search_params, get_soh_based_on_ods_results


"""
    所有电芯 soh 运行主函数，输入参数包括：
        采样真实数据 df_real, 采样频率 freq, 电池额定容量 full_cap, 
        箱内全部电池的电压列名 all_volt_names - list,
        辨识-电压计算-loss的 .so 文件 so_path - str,
        
        需要进行局部暴力求解的电芯号（输入的最差电芯）cells_for_ods - int,
        
        change_param_num - 在参数列顺序固定的前提下，改变前 change_param_num 列进行局部搜索
        range_ratio_max - 搜索范围的圆半径
        
        文件保存的路径 sup_out_url, 
        文件保存标志 save_flag
        画图标志 if_plot
"""
def run_soh(df_real, freq, full_cap, battery_type, all_volt_names, so_path, cells_for_ods,
            # change_param_num=24, range_ratio_max=0.2,
            change_param_num=24, range_ratio_max=0.5,
            sup_out_url='', save_flag=False, if_plot=False):    # 默认先不画图，要画图需要设置画图保存路径 

    """
        # [step 0] - 全方向搜索需要 变化的 列名 - 可以等到 全方向 搜索 这一步 再 确定
    """
    # change_columns = ...
            
    """
        # [step 0] - 参数 辨识 的输入 收集
    """
    sipp_obj = Str_Info_for_Procure_Param(so_file_url = so_path,
                                            battery_type = battery_type,
                                            loss_func_str = 'mae',   # 损失函数类型
                                            
                                            # 参数辨识 用到
                                            iden_task_num_Max = 1000,     # 本次参数辨识的任务数 task_num
                                            iden_result_len_threshold = 200,     # 参数辨识输出的结果 最少要留 的 参数行数
                                            each_cell_inden_cnt_Max = 10,    # 单节电池最多连续辨识的 次数 - 为了满足辨识结果 > iden_result_len_threshold
                                            
                                            # 归一化球坐标 用到
                                            select_rows_from_param_iden = 50,   # 从参数辨识结果中 选择这些参数 去 归一化求距离
                                            # 全方向搜索 用到
                                            select_rows_for_ods = 2)   # 在 全方向搜索 ods 时，根据距离 选择的参数行数 - 必须 > 1，否则全部行参数都会选上
    
    """
        # [step 0] - 输出变量初始化
    """
    soh_output = SOH_Output(cell_index = 0, 
                            cell_list = [],
                            soh_list = [],
                            soh_std_list = [],
                            param_with_soh_dict = {},
                            # 统计每节电池辨识了多少次
                            count_list_of_each_cell = [])
    
    # 逐节电芯 开始计算

    log.logger.info(f'Start run_soh, battery type is {battery_type}')
    for i, vol_name in enumerate(all_volt_names):
        
        """
        # [step 0] - 读取真实电压数据
        """
        
        dufv_obj = Data_Used_to_Fit_Volt(df_real = df_real,     # 真实电压直接输入
                                        freq = freq,
                                        full_cap = full_cap,
                                        conversion_efficiency = 1,
                                        vol_name = vol_name)    # 电压的列名 str - 即第几节电芯
        
        log.logger.debug(f'dufv_obj.df_real:{dufv_obj.df_real}')
        # assert 0
        
        cell_no = vol_name.split('-')[-1]
        soh_output.cell_list.append(cell_no)
        soh_output.cell_index = cell_no     # 表示当前处理的电芯序号
        
        # 直接传入 soh_output, 函数就不加返回值了
        steps_each_cell(dufv_obj, sipp_obj, cells_for_ods, soh_output, 
                        change_param_num, range_ratio_max,
                        sup_out_url, save_flag=save_flag, if_plot=if_plot)    # 默认先不画图，要画图需要设置画图保存路径
        
        log.logger.debug(f'The calculation of the {i}-th battery soh: {soh_output}')
    
    log.logger.info(f'The calculation of soh is completed \tsoh_output : {soh_output}')
    # 各节电芯循环结束后，返回计算结果    
    return soh_output
    

def steps_each_cell(dufv_obj, sipp_obj, cells_for_ods, soh_output,
                    change_param_num, range_ratio_max,
                    sup_out_url, save_flag=False, if_plot=False):      # 结果保存 - 这是内部参数
    
    """
        # [step 1] - 参数 辨识 + 选择参数
    """
    time_start = time.time()
    parameters_iden, param_bound = param_identify_then_filter(dufv_obj, sipp_obj, max_iter=500)
    log.logger.debug(f'soh_output : {soh_output}')
    log.logger.debug('=== 参数辨识完成 ===')
    log.logger.debug(f'parameters_iden : {parameters_iden}')
    log.logger.debug(f"parameters_iden : {parameters_iden[['loss', 'Area', 'SOH']]}")
    log.logger.debug(f'param_bound : {param_bound}')
    
    log.logger.debug(f"parameters_iden : {parameters_iden['SOH']}")
    log.logger.debug(f"parameters_iden : {parameters_iden['SOH'].max(), parameters_iden['SOH'].min()}")

    # print('=== 参数辨识完成 ===')
    # print(f'parameters_iden : {parameters_iden}')
    # print(f"parameters_iden : {parameters_iden[['loss', 'Area', 'SOH']]}")
    # print(f'param_bound : {param_bound}')
    
    # print(f"parameters_iden : {parameters_iden['SOH']}")
    # print(f"parameters_iden : {parameters_iden['SOH'].max(), parameters_iden['SOH'].min()}")
    """
        # [step 2] - 大部分电池可直接通过 SOH 分布来计算 SOH 值
    """
    param_after_soh_fit, soh_std = run_soh_distribution(parameters_iden, save_flag=save_flag, if_plot=if_plot, soh_hist_url=os.path.join(sup_out_url, dufv_obj.vol_name+'.png'), return_param=True)
    soh_this_cell = param_after_soh_fit['SOH'].iloc[0]
    loss = param_after_soh_fit['loss'].iloc[0]
    
    soh_output.soh_list.append(soh_this_cell)
    soh_output.soh_std_list.append(soh_std)
    soh_output.param_with_soh_dict[soh_output.cell_index] = param_after_soh_fit
    log.logger.debug(f'soh_output : {soh_output}')
    # mayan add to print 
    time_end = time.time()
    print(f"before ods time:{round(time_end-time_start,4)}s  SOH：{round(soh_output.soh_list[-1],1)} loss:{round(loss,2)}")
    
    """
        # [step 3] - 有些 电池 需要 局部搜索，选择参数后开启局部搜索
    """
#    if str(cells_for_ods) == soh_output.cell_index:
    if 0:   # 全部电池都测试一次
        log.logger.debug(f'第 {soh_output.cell_index} 节电池进行 全方向搜索')
        # print(f'第 {soh_output.cell_index} 节电池进行 全方向搜索')
    
        """
        # [step 3 - 1] - 球坐标算距离，要注意输入的参数是否根据 loss 筛选 前 _ 行
        # 【重要】 参数赋值 时需要注意 选取这些列 - cuda 指定了最多只能有这些列，同时提前把 loss 取出来
        # [step 3 - 2] - change_columns 参数的上下限
        """
        
        # 这里返回的 parameters_with_all_cols 是包含所有列的
        parameters_with_all_cols, change_columns, params_up_low_list = params_range_search(parameters_iden, param_bound, sipp_obj, change_param_num, range_ratio_max=range_ratio_max)
        log.logger.debug(f"parameters_with_all_cols : {parameters_with_all_cols}")
        log.logger.debug(f"change_columns : {change_columns}")
        log.logger.debug(f"params_up_low_list : {params_up_low_list}")
        # print(f"parameters_with_all_cols : {parameters_with_all_cols}")
        # print(f"change_columns : {change_columns}")
        # print(f"params_up_low_list : {params_up_low_list}")
        
        """
        # [step 3 - 3] - 全范围搜索 ods, 输出的 grr_obj 包含 搜索筛选后的参数
        """
        start_time = time.time()
        log.logger.debug(f'===== cell {cells_for_ods} 开始 全方向搜索啦 =====')
        # print(f'===== cell {cells_for_ods} 开始 全方向搜索啦 =====')
        
        grr_obj = ods_search_params(parameters_with_all_cols, change_columns, params_up_low_list, dufv_obj, sipp_obj)
        
        end_time = time.time()
        log.logger.debug(f'=== ods run time === {end_time - start_time} s')
        log.logger.debug(f"loss_before_ods : {parameters_with_all_cols['loss'].tolist()}")
        # print(f'=== ods run time === {end_time - start_time} s')
        # print(f"loss_before_ods : {parameters_with_all_cols['loss'].tolist()}")
        
        parameters_to_cal_soh = get_soh_based_on_ods_results(grr_obj)
        soh_after_ods = parameters_to_cal_soh['SOH'].mean()
        loss = parameters_to_cal_soh['loss'].mean()
        
        # 将全局搜索后的参数求出的 soh 值作为该节电池的 soh，替换上面的 soh_this_cell
        soh_output.soh_list[-1] = soh_after_ods
        soh_output.soh_std_list[-1] = parameters_to_cal_soh['SOH'].std()
        soh_output.param_with_soh_dict[soh_output.cell_index] = parameters_to_cal_soh
        
        log.logger.debug(f'{soh_output.cell_index} 电芯 全方向搜索 结束 ')
        log.logger.debug(f'全方向搜索 ods 完成，soh_after_ods 为 {soh_after_ods}')
        log.logger.debug(f'全方向搜索 ods 完成，soh_output : {soh_output}')
        # print(f'{soh_output.cell_index} 电芯 全方向搜索 结束 ')
        # print(f'全方向搜索 ods 完成，soh_after_ods 为 {soh_after_ods}')
        # print(f'全方向搜索 ods 完成，soh_output : {soh_output}')

        # # 画图临时加的
        # from ods_search_params import plot_volt_on_ods_results
        # plot_volt_on_ods_results(grr_obj, dufv_obj, sipp_obj)
        
        # assert 0
    
    # mayan add to print 
    time_end = time.time()
    print(f"after ods time:{round(time_end-time_start,4)}s  SOH：{round(soh_output.soh_list[-1],1)} loss:{round(loss,2)}")
    log.logger.info(f'{soh_output.cell_list} 电芯 计算完成 \n{soh_output.cell_index} soh:{soh_output.soh_list[-1]}')
