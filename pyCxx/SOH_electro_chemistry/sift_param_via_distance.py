"""
    多组参数归一化 + 部分 log 化后求 距离，选择距离特定参数 最远的几组参数，并对应计算这些参数 全方向搜索的边界
"""

import numpy as np
import pandas as pd

"""
    log.logger.debug("用于打印日志")
"""
from tool_utils.pvlog import Logger, level_SOH_electro
log = Logger('./logs/SOH_electro.log', level=level_SOH_electro)

"""
    call_electrochem_identify : 参数辨识相关的接口，也包含参数的列名获取，不同行 DataFrame 合并
"""
from .call_electrochem_identify import reset_param_columns_to_fit_vol, get_param_name_list

# '''
#     将参数固定为只取这些列，原因是在 GPU 上进行全方向搜索时，输入参数限制最多只能有这 28 列，且顺序固定
#     而且传入 GPU 代码并行计算 损失时必须按照这个固定列
# '''
# def reset_param_columns_to_fit_vol(df_param):
#     # columns = ["Area","L_pos","epss_pos","L_neg","epss_neg","L_sep","epsl_sep","csmax_pos","csmax_neg",
#     #         "rp_pos","rp_neg","Ds_pos","Ds_neg","epsl_pos","epsl_neg","ce_0","tc","m_pos","m_neg",
#     #         "cs0_pos","cs0_neg","ks_pos","ks_neg","resist","e_D_p","e_D_n","e_mref_p","e_mref_n",
#     #         ]  
#     columns = get_param_name_list()
#     df_param = df_param[columns]

#     return df_param


'''
    将一列 Series 转为 一行 DataFrame
'''
def convert_Series_to_DataFrame(out_param_df, chosen_index=-1):
    if type(out_param_df) not in [pd.DataFrame]:
        parameters = out_param_df.to_frame().T
    elif type(out_param_df) in [pd.DataFrame]:
        parameters = out_param_df.iloc[chosen_index].to_frame().T   # 选择 某一行，然后转为 DataFrame
    return parameters



# 部分log化
class Analyze_Param(object):
    
    def __init__(self, df_params, column_names, param_bound, points_num, part_col_log=True, norm_based_on_bound=False, flag='far'):
        self.df_params = df_params
        self.column_names = column_names
        self.param_bound = param_bound
        
        self.points_num = points_num
        self.part_col_log = part_col_log
        self.norm_based_on_bound = norm_based_on_bound
        self.flag = flag
    
    def ap_obj_run(self):
        # 整个过程 不涉及 不同行 调换
        df = self.process_param_before_cal_distance()
        chosen_index = self.params_at_furthest_distance(df)
        
        df_out = self.df_params.iloc[chosen_index].reset_index(drop=True)       # 包含 loss 列
        
        return df_out
        
    def process_param_before_cal_distance(self):
        df = self.df_params[self.column_names].copy()
        log.logger.debug(f'df.columns : {df.columns}')
        # print(f'df.columns : {df.columns}')
        if self.part_col_log:
            df = self.get_part_cols_log(df)     # 部分参数需要log处理

        df = self.normalize_df(df)
    
        return df
    
    def get_part_cols_log(self, df):
        # 需要log处理的参数列名
        log_col_list = ['L_sep', 'rp_pos', 'rp_neg', 'Ds_pos', 'Ds_neg', 'm_pos', 'm_neg']
        assert all(elem in self.column_names for elem in log_col_list), 'log 处理的 列名 不在 原参数列里'
        
        for col in df.columns:
            if col in log_col_list:
                df[col] = np.log(df[col])
        
        return df
    
    def normalize_df(self, df):      # 归一化
    
        column_names = self.column_names[:24]       # 只要前 24 个参数 - 不要温度相关的 那4个 参数
       
        # 将所有参数归一化，用辨识出的数据的最大和最小值
        normalize = lambda x: (x-x.min()) / (x.max()-x.min())
        df[column_names] = df[column_names].apply(normalize)
    
        df.reset_index(drop=True, inplace=True)
    
        return df
    
    def params_at_furthest_distance(self, df):
        min_loss_index = 0
        # 获取第一行的数据（转换为NumPy数组） - loss 最小的 参数
        first_row = df.iloc[min_loss_index].values  
        
        # 使用广播机制计算每行与第一行的差异  
        diff = df.values - first_row  
        # 计算每行的欧氏距离  
        # 注意：np.linalg.norm的axis=1参数是沿着行计算，即对每个子数组（每行）计算范数  
        distances = np.linalg.norm(diff, axis=1)  

        points_num = self.points_num - 1    # 留出来一个参数是 给 loss 最小的那个参数
        if self.flag in [1, 'big', 'far']:
            chosen_index = distances.argsort()[-points_num:][::-1]       # 注意负号, 距离最大的点 从大到小 排序
        elif self.flag in [0, 'small', 'close']:
            chosen_index = distances.argsort()[:points_num]        # 距离最小的点的索引 从小到大 排序
        
        # 在 末尾插入 min_loss_index
        chosen_index = np.append(chosen_index, min_loss_index)
    
        return chosen_index


class Params_Search_Range(object):
    
    """
        输入参数：
            params 是输入的参数
            change_columns 是修改的列
            up_low_range 是输入的参数的上下区间 - 来自参数辨识
            range_ratio_max 是参数所搜索的最大区间（比如参数的 5%)）
            rows_to_make_params 是生成参数的行数
    """
    def __init__(self, params, up_low_range, change_columns, range_ratio_max):
        self.params = params
        self.up_low_range = up_low_range
        self.change_columns = change_columns
        self.range_ratio_max = range_ratio_max
    
    """
        基于输入的参数 生成 搜索区间 - 每行参数有自己单独的范围
    """
    def make_params_up_low_of_this_row(self, intersection=True):    # intersection 表示 取 自定义的范围 & 之前参数字典里范围的交集

        # 两行用来存 up & low - self.params 是一行 DataFrame
        params_up_low = self.params.copy()
        params_up_low = params_up_low._append(params_up_low, ignore_index=True)     # 两行用来存 up & low - 注意不同的 Python 版本 _append 要换成 append
        #params_up_low = params_up_low.append(params_up_low, ignore_index=True)     # 两行用来存 up & low - 注意不同的 Python 版本 _append 要换成 append
        
        # 生成上下区间 - 针对 change_columns
        param_change_range_ratio = self.range_ratio_max
        params_up_low_of_change_columns = params_up_low[self.change_columns].copy()
        
        params_up_low_of_change_columns.iloc[0] *= (1 + param_change_range_ratio)        # 上限
        params_up_low_of_change_columns.iloc[1] *= (1 - param_change_range_ratio)       # 下限
        
        params_up_low[self.change_columns] = params_up_low_of_change_columns
        
        # 不变的列上下区间 相同
        cols_not_changed = list(set(self.params.columns) - set(self.change_columns))
        params_up_low[cols_not_changed] = self.params[cols_not_changed].iloc[0]    # 建议只输入一行参数（不然每行参数的固定列上下限值就得变），这里取 输入参数 第一行的值
        
        params_up_low.rename(index={0: 'up', 1: 'low'}, inplace=True)   # index 更名

        log.logger.debug(f'params_up_low_of_change_columns: {params_up_low_of_change_columns}')
        log.logger.debug(f'params_up_low : {params_up_low}')
        # print(f'params_up_low_of_change_columns: {params_up_low_of_change_columns}')
        # print(f'params_up_low : {params_up_low}')
        # print(f"params_up_low: {params_up_low.loc['up']}")
        
        if intersection:
            log.logger.debug(f'self.up_low_range: {self.up_low_range}')
            # print(f'self.up_low_range: {self.up_low_range}')
            # print(f"self.up_low_range: {self.up_low_range.loc['up']}")
            
            # params_up_low 是固定 变化比例生成的边界
            result_df = self.intersect_params_up_low(params_up_low)
            log.logger.debug(f'result_df: {result_df}')
            # print(f'result_df: {result_df}')
        else:
            result_df = params_up_low.copy()
            
        # assert 0
        return result_df
    
    """
        注意：有三个参数的上下限需要根据数据更新，这里是没有更新的 这几个参数上下界
    """
    def intersect_params_up_low(self, params_up_low):   
        columns = self.up_low_range.columns
        # df_a = self.up_low_range[self.change_columns].copy()    # 注意 up 为 index = 0; low 为 index = 1
        # df_b = params_up_low[self.change_columns].copy()
        
        df_a = self.up_low_range[columns].copy()    # 注意 up 为 index = 0; low 为 index = 1
        df_b = params_up_low[columns].copy()
        
        # 将DataFrame的值转换为NumPy数组  
        values_a = df_a.values  
        values_b = df_b.values      # 有几个不变的参数，其上下边界一样，肯定是位于原给定的上下边界 之间的
        
        log.logger.debug(f'df_a: {df_a}')
        log.logger.debug(f'df_b: {df_b}')
        # print(f'df_a: {df_a}')
        # print(f'df_b: {df_b}')
        # print(f'values_a: {values_a}')
        # print(f'values_b: {values_b}')
        
        # 使用 NumPy 的 minimum 函数找到对应元素的最小值  
        result_min = np.minimum(values_a, values_b)     # 上限 取 小值
        result_max = np.maximum(values_a, values_b)     # 下限 取 大值 
        
        result_array = np.asarray([result_min[0], result_max[-1]])
        # result_array = np.minimum(values_a, values_b)  
        log.logger.debug(f'result_array: {result_array}')
        # print(f'result_array: {result_array}')
        
        # 将结果数组转换回DataFrame  
        result_df = pd.DataFrame(result_array, index=df_a.index, columns=df_a.columns)  

        return result_df    
        

"""
    全方向 搜索时 参数的上下限区间
"""
def get_param_search_range(params, up_low_range, change_columns, range_ratio_max=0.2):
    
    params_up_low_list = []
    # 每行参数去生成 其 对应的 范围
    for i, params_this_row in params.iterrows(): 
        params_this_row = convert_Series_to_DataFrame(params_this_row)
        
        psr_obj = Params_Search_Range(params_this_row, up_low_range, change_columns, range_ratio_max)
        params_up_low = psr_obj.make_params_up_low_of_this_row(intersection=True)
        
        params_up_low_list.append(params_up_low)
    
    return params_up_low_list


def params_range_search(parameters_iden, param_bound, sipp_obj, change_param_num, range_ratio_max=0.2):
    
    # [step 3 - 1] - 球坐标算距离，选择 MAE 在阈值内的参数中，距离最远的几组参数，作为全方向搜索的输入
    parameters_of_aimed_cols = parameters_iden.iloc[:sipp_obj.select_rows_from_param_iden].reset_index(drop=True)
  
    ap_obj = Analyze_Param(parameters_of_aimed_cols, get_param_name_list(), param_bound, sipp_obj.select_rows_for_ods, part_col_log=True, norm_based_on_bound=False, flag='far')
    
    parameters_with_all_cols = ap_obj.ap_obj_run()
    log.logger.debug(f"parameters : {parameters_with_all_cols}")
    log.logger.debug(f"parameters : {parameters_with_all_cols[['loss', 'Area', 'SOH']]}")
    # print(f"parameters : {parameters_with_all_cols}")
    # print(f"parameters : {parameters_with_all_cols[['loss', 'Area', 'SOH']]}")
    
    # 【重要】 参数赋值 时需要注意 选取这些列 - cuda 指定了最多只能有这些列，同时提前把 loss 取出来
    parameters = reset_param_columns_to_fit_vol(parameters_with_all_cols)
    change_columns = parameters.columns[:change_param_num]      # 要搜索的参数列名 - 前 change_param_num(usually 24)
    
    log.logger.debug(f'change_columns : {change_columns}')
    # print(f'change_columns : {change_columns}')
    
    # [step 3 - 2] - change_columns 参数的上下限
    # 这里输入的 parameters 与 param_bound 列数一样，只有 28 列；并且输出结果只有 change_columns 列对应的上下限是不一样的
    params_up_low_list = get_param_search_range(parameters, param_bound, change_columns, range_ratio_max=range_ratio_max)
    log.logger.debug(f'params_up_low_list : {params_up_low_list}')
    # print(f'params_up_low_list : {params_up_low_list}')
    
    # return parameters_with_all_cols, parameters, change_columns, params_up_low_list
    return parameters_with_all_cols, change_columns, params_up_low_list     # 选择固定列的 parameters 可以不返回