import numpy as np
import pandas as pd

from dataclasses import dataclass, fields

from typing import get_origin, get_type_hints

import copy  


"""
    迭代过程中保存  拟合电压 vol_out_df, 参数 out_param_df, 损失 out_rmse_list / mse_df
"""
@dataclass
class Volt_Fit_and_Loss:
    
    vol_out_df : pd.DataFrame
    out_param_df : pd.DataFrame
    
    mse_df : pd.DataFrame
    out_rmse_list : list
    
    def reset_some_attributes(self):    # out_param_df 不清空
        self.vol_out_df = pd.DataFrame() 
        self.mse_df = pd.DataFrame()  
        
        self.out_rmse_list = []
      
    # copy.copy 是浅拷贝
    def __copy__(self):  
        # 使用 vars() 获取当前对象的所有属性及其值  
        attributes = vars(self)  
        # 创建一个新的 MyDataClass 实例，并通过解包字典的方式传入属性值  
        # 这里使用字典解包（**）语法来传递属性到 MyDataClass 的构造函数中  
        return Volt_Fit_and_Loss(**{k: copy.copy(v) if isinstance(v, (list, dict)) else v for k, v in attributes.items()})  
    
    def __deepcopy__(self, memo):  
        # 使用 vars() 获取当前对象的所有属性及其值  
        attributes = vars(self)  
        # 使用 copy.deepcopy() 进行深拷贝  
        return Volt_Fit_and_Loss(**{k: copy.deepcopy(v, memo) if isinstance(v, (list, dict)) else v for k, v in attributes.items()}) 


"""
    迭代过程中保存  参数上下限 params_up_low, 需要计算的列名change_columns, 
    并行处理时一次的并行批次 batch_size, 搜索求解的最大轮次数 max_iter, 迭代停止的电压拟合误差比较基准值 iterative_cutoff_rmse_critical_value
"""
@dataclass
class Omni_Directions_Several_Steps:
    params_up_low : pd.DataFrame
    params_up_low_list : list 
    
    change_columns : list
    
    batch_size : int
    max_iter : int
    iterative_cutoff_rmse_critical_value : float    # 迭代截止的 rmse / mad 基准值
    
    # gpu_category : str
    
    # base_change_step 变量在，则 该方法也要在
    def reset_some_attributes(self):  
        # self.base_change_step = []    # 这个是否需要注释，结合代码再定
        pass
        
"""
    保存 电池相关数据 + 真实采样数据，包括： 
    真实电压值 df_real, 采样频率 freq, 电池额定容量 full_cap, 库伦效率 conversion_efficiency,
    此时该节电池的电压列名 vol_name
"""       
@dataclass
class Data_Used_to_Fit_Volt:

    df_real : pd.DataFrame

    freq : int
    full_cap : float
    conversion_efficiency : float
    
    vol_name : str  # 电压的列名
    
    # change_cnt : int    # 参数修改的次数 - 存储方便给列名


"""
    迭代过程中保存 每步的输出结果，包括： 
    初始参数与最后所选参数 all_params_out_df, 迭代轮次 iter_cnt, 迭代期间参数记录 param_df_record, 迭代期间loss最小的那组参数 param_at_min_rmse
"""  
@dataclass
class Gradient_Result_Record :

    all_params_out_df : pd.DataFrame
    
    iter_cnt : list
    
    param_df_record : list
    rmse_record : list
    param_at_min_rmse : list

"""
    参数辨识需要的信息，包括： 
    堆簇箱 sta_clu_box, 场站编号 station_code, 辨识-电压计算-loss的.so文件 iden_so_file_url, 参数辨识最大任务数 iden_task_num_Max, 从参数辨识结果中选择的参数行数 select_rows_from_param_iden
"""  
@dataclass
class Str_Info_for_Procure_Param :

    # sta_clu_box : str
    # station_code : str
    
    so_file_url : str

    battery_type : str
    loss_func_str : str
    
    iden_task_num_Max : int
    iden_result_len_threshold : int
    each_cell_inden_cnt_Max : int
    
    select_rows_from_param_iden : int
    select_rows_for_ods : int
    
    
@dataclass
class SOH_Output :
    # 输出变量
    cell_index : int
    
    cell_list : list
    
    soh_list : list
    soh_std_list : list
    
    param_with_soh_dict : dict
    
    # 统计每节电池辨识了多少次
    count_list_of_each_cell : list
    
    
"""
    获取真实数据的类 - 在参数辨识 / 全方向搜索时都用到了，统一写在这里，到时调用即可
"""        
class Real_Data_Info():
    
    def __init__(self, dufv_obj):
        self.dufv_obj = dufv_obj

    def get_real_data_info(self):

        self.time_arr = self.dufv_obj.df_real['time [s]'].values.astype(np.float32)      # 0, 10, 20, ...
        self.current_arr = self.dufv_obj.df_real['current'].values.astype(np.float32)
        self.voltage_arr = self.dufv_obj.df_real[self.dufv_obj.vol_name].values.astype(np.float32)    # 电压列名
        self.temperature_arr = self.dufv_obj.df_real['temp (℃)'].values.astype(np.float32) + 273.15      # 温度的均值列 + 273.15
        
        if self.temperature_arr.max() < 100:
            # 转化为开尔文温度
            self.temperature_arr += 273.15
            
        self.soc_arr = self.dufv_obj.df_real['soc'].values      #  soc 不要除以 100， 三元电池代码里 自己处理了
        # self.df_real_length = len(time_arr)      # 原始数据的长度
        
           

    