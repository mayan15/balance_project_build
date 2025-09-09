import numpy as np
import pandas as pd

import ctypes

import random

"""
    storage_database : 所需数据类
"""
from .storage_database import Real_Data_Info

"""
    electrochem_mpi : mpi 调用 .so 文件的接口
"""
from .electrochem_mpi import MpiEC

"""
    电化学参数辨识 Python 接口类，调用 cuda 代码
    输入参数： 
    - 以下四个参数是辨识要输入的原始数据
    time - np.ndarray
    current - np.ndarray
    voltage - np.ndarray
    temperature - np.ndarray
    - soc 是 根据电池的 输入 soc 来更新参数边界，并确定初始的参数点 p0
    soc - np.ndarray
    - 以下几个参数是电池的基本信息
    rated_capacity - float （额定容量）
    battery_type - str （电池类型）
    
    so_file_str - .so 文件 （调用 cuda / musa 程序）
    
    loss_func - str （损失函数字符串）
    max_iter - int （最大迭代次数）
    task_num - int （辨识任务数）

"""
class Elecrochemical_platform:
     
    def __init__(self, time, current, voltage, temperature, soc, rated_capacity, 
                 battery_type, 
                 so_file_str, loss_func,
                 param_name_list, lumped_param_name_list): 
        
        self.time = np.array(time, dtype=np.float32)
        self.current = np.array(current, dtype=np.float32)
        self.voltage = np.array(voltage, dtype=np.float32)
        self.temperature = np.array(temperature, dtype=np.float32)   # temperature 记得 + 273.15
        
        self.soc = soc  # soc 不要除以 100， 三元电池代码里 自己处理了
        
        self.rated_capacity = float(rated_capacity)
       
        self.battery_type = battery_type
        self.so_file_str = so_file_str
        # self.lib = ctypes.CDLL(so_file_str)       # 调用的计算接口内部 将 .so 文件转换为 DLL 库
        
        self.loss_func = loss_func
        # self.max_iter = max_iter      # 以下两个变量直接在调用的方法中作为参数传入
        # self.task_num = task_num
        
        self.column_names = lumped_param_name_list + param_name_list  # 添加 "loss" 列标题到最前面
        
    
    def ep_obj_run(self, particle_num=64, para_num=28, max_iter=500, task_num=500, all_gather=1):
        # 预处理数据 - 分电池类型去处理
        self.pre_process_data()
        # 获取边界和初始参数 - 分电池类型去处理
        self.update_param_bound(get_bound_flag=True)    # 要更新参数边界    
        
        # 开始辨识并输出结果
        print('Start param identify !')
        # 09-06
        idf_result = self.get_parameters(particle_num=particle_num, para_num=para_num, max_iter=max_iter, task_num=task_num, all_gather=all_gather)

        print(f'ep_obj_run()运行完毕，参数 辨识 完成 ')
        # print(f'{self.vol_name} 辨识 完成 ')
        # self.parameters_iden = {self.vol_name: idf_result.copy()}
        self.parameters_iden = idf_result.copy()

    # 这个函数的功能其实在 其他类中已经处理过了，这里是为了多保险
    def pre_process_data(self):
        
        self.data_length = len(self.time)
        if self.temperature.max() < 100:
            # 转化为开尔文温度
            self.temperature += 273.15
    
    def update_param_bound(self, get_bound_flag=True):
        if self.battery_type in ['LFP', 'lfp']:
            self.cut_voltage_low_limit = 2.5
            self.cut_voltage_up_limit = 3.65
            # LFP
            electro_obj = Electrochemical_LFP(self.soc, self.rated_capacity)
            
        elif self.battery_type in ['NCM', 'ncm', 'NMC', 'nmc']:
            self.cut_voltage_low_limit = 2.5
            self.cut_voltage_up_limit = 4.2
            # NCM
            electro_obj = Electrochemical_NCM(self.soc, self.rated_capacity)
        
        else:
            raise Exception('Wrong Battery Type !')
        
        if get_bound_flag:
            # electro_obj 可能来自不同的 电池类型 数据类
            electro_obj.get_up_low_bound()
            
            self.p0 = np.array(electro_obj.p0, dtype=np.float32)
            self.upbound = np.array(electro_obj.upbound, dtype=np.float32)
            self.lowbound = np.array(electro_obj.lowbound, dtype=np.float32)
            
            # 参数的上下边界 - 部分参数的边界根据 soc / capacity 更新过
            self.params_up_low_bound_dict = electro_obj.parameters_dict

    
    """
        param_name_list - list （参数辨识后参数的列名，与 para_num 的元素个数要相等）
    """
    def get_parameters(self, particle_num, para_num, max_iter, task_num, all_gather):
        # assert para_num == len(param_name_list), "param_name_list 元素个数 != para_num"
        
        solver = MpiEC(all_gather)
        solver.data_init(self.so_file_str, self.time, self.current, self.voltage, self.temperature, para_num)
        
        print(f'self.loss_func : {self.loss_func}')
        
        arr_2d = solver.get_pso_result(self.p0, self.upbound, self.lowbound, 
                                       task_num=task_num, 
                                       particle_num=particle_num, para_num=para_num, 
                                       max_iter=max_iter, 
                                       capacity_real=self.rated_capacity, 
                                       cut_voltage_up=self.cut_voltage_up_limit, 
                                       cut_voltage_low=self.cut_voltage_low_limit, 
                                       loss_func=self.loss_func, battery_type=self.battery_type)  
        
        parameters = self.post_process_parameters(arr_2d, param_filter_flag=True)
        
        return parameters
    
    def post_process_parameters(self, arr_2d, param_filter_flag=True):
       
        # 数据保存
        parameters = pd.DataFrame(arr_2d, columns=self.column_names)  # 将 numpy 数组转换为 pandas DataFrame，并设置列标题
        # 计算 SOH
        parameters['SOH'] = parameters['total_capacity (Ah)'] / self.rated_capacity * 100
        
        if param_filter_flag:
            parameters = self.parameters_filter_func(parameters)
        
        return parameters
    
    def parameters_filter_func(self, parameters):

        # print(f' == 00 == Iden parameters: {parameters}')
        
        parameters = parameters[(parameters >= 0).all(axis=1)].reset_index(drop=True)
        # print(f' == 11 == Iden parameters: {parameters}')
        
        parameters = parameters[(parameters["current_energy (KWh)"] > 0)]
        # print(f' == 22 == Iden parameters: {parameters}')
        
        capacity_multi_coeff = 1.1
        parameters = parameters[parameters["total_capacity (Ah)"] <= self.rated_capacity * capacity_multi_coeff]
        # print(f' == 33 == Iden parameters: {parameters}')
        
        if self.loss_func == 'mae':   # loss 不可以太大
            parameters = parameters[parameters["loss"] <= 15] 
        
        if self.loss_func == 'mse':   # loss 不可以太小
            parameters = parameters[parameters["loss"] >= 10] 

        # print(f' == 44 == Iden parameters: {parameters}')
        
        return parameters

    
class Electrochemical_NCM:
        
    def __init__(self, soc, rated_capacity):
        self.soc = soc
        self.rated_capacity = rated_capacity
        
        self.parameters_dict = {'Area': {'val': 0.1027, 'up': 0.123, 'low': 0.0822},  # parameters for 5Ah
                                'L_pos': {'val': 7.56e-05, 'up': 9.1e-05, 'low': 6.1e-05},
                                'epss_pos': {'val': 0.665, 'up': 0.798, 'low': 0.532},
                                'L_neg': {'val': 8.52e-05, 'up': 0.00010224, 'low': 6.816e-05},
                                'epss_neg': {'val': 0.75, 'up': 0.9, 'low': 0.6},
                                'L_sep': {'val': 1.2e-05, 'up': 1.44e-05, 'low': 9.6e-06},
                                'epsl_sep': {'val': 0.47, 'up': 0.564, 'low': 0.376},
                                # 'csmax_pos': {'val': 63104.0, 'up': 65000, 'low': 61000}, # range
                                # 'csmax_neg': {'val': 33133.0, 'up': 35000, 'low': 32000},
                                'csmax_pos': {'val': 63104.0, 'up': 75724.8, 'low': 50483.2},
                                'csmax_neg': {'val': 33133.0, 'up': 36133.0, 'low': 30133.0},
                                'rp_pos': {'val': 5.22e-06, 'up': 6.26e-06, 'low': 4.18e-06},
                                'rp_neg': {'val': 5.86e-06, 'up': 7.032e-06, 'low': 4.688e-06},
                                'Ds_pos': {'val': 4e-15, 'up': 4.8e-15, 'low': 3.2e-15},
                                'Ds_neg': {'val': 3.3e-14, 'up': 3.96e-14, 'low': 2.64e-14},
                                'epsl_pos': {'val': 0.335, 'up': 0.402, 'low': 0.268},
                                'epsl_neg': {'val': 0.25, 'up': 0.3, 'low': 0.2},
                                'ce_0': {'val': 1000.0, 'up': 1200.0, 'low': 800.0},
                                'tc': {'val': 0.2594, 'up': 0.31128, 'low': 0.21},
                                'm_pos': {'val': 3.42e-06, 'up': 4.104e-06, 'low': 2.736e-06},
                                'm_neg': {'val': 6.48e-07, 'up': 7.776e-07, 'low': 5.184e-07},
                                'cs0_pos': {'val': 17038.0, 'up': 20445.6, 'low': 13630.4},
                                'cs0_neg': {'val': 29866.0, 'up': 35839.2, 'low': 23892.8},
                                'ks_pos': {'val': 0.18, 'up': 0.216, 'low': 0.144},
                                'ks_neg': {'val': 215, 'up': 258.0, 'low': 172.0},
                                'resist': {'val': 0.0001, 'up': 0.001, 'low': 0.0},
                                # 下面四列 是直接从 LFP 复制来的，后续需要确定具体值 - 0914
                                'e_D_p': {'val': 47000.0, 'up': 235000.0, 'low': -47000.0},
                                'e_D_n': {'val': 30300.0, 'up': 60600.0, 'low': -30300.0},
                                'e_mref_p': {'val': 39570, 'up': 395700, 'low': -39570},
                                'e_mref_n': {'val': 35000, 'up': 350000, 'low': -35000}}

    """
    通过ocv -- 映射 soc
    """
    def get_parameters_range(self):  # 注意可能存在的bug 初始电压值非ocv值
        # target_capacity : nmc-rated_capacity
        # soc: soc_list 0 - 100
        
        self.parameters_dict['Area']['low'] = self.parameters_dict['Area']['val'] * self.rated_capacity / 5 * 0.8
        self.parameters_dict['Area']['up'] = self.parameters_dict['Area']['val'] * self.rated_capacity / 5 * 1.2
        self.parameters_dict['cs0_neg']['low'] = self.parameters_dict['csmax_neg']['val'] * max(self.soc[0] / 100 * 0.85 - 0.01,
                                                                                    0.04)  # + 0.045 # 0.9 - 0.05 delta sto = 0.05 / 5%
        self.parameters_dict['cs0_neg']['up'] = self.parameters_dict['csmax_neg']['val'] * min(self.soc[0] / 100 * 0.85 + 0.01,
                                                                                    0.92)  # +0.055
        self.parameters_dict['cs0_pos']['low'] = self.parameters_dict['csmax_pos']['val'] * max(0.85 - self.soc[0] / 100 * 0.6 - 0.05,
                                                                                    0.2)  # 0.25 - 0.84
        self.parameters_dict['cs0_pos']['up'] = self.parameters_dict['csmax_pos']['val'] * min(0.85 - self.soc[0] / 100 * 0.6 + 0.05, 0.9)

    def get_up_low_bound(self):
        
        # 准备参数辨识的上下限阈值
        self.get_parameters_range()  # , sto_neg) #) # soc_first) #
        upbound = [self.parameters_dict[key]['up'] for key in self.parameters_dict.keys()]
        lowbound = [self.parameters_dict[key]['low'] for key in self.parameters_dict.keys()]
        
        print('辨识Area upbound', upbound[0])
        print('辨识Area lowbound', lowbound[0])
        
        # 初始化需要辨识的参数(取上下限阈值的平均)
        p0 = 0.5 * (np.array(upbound) + np.array(lowbound))
        
        self.p0 = np.array(p0, dtype=np.float32)
        self.upbound = np.array(upbound, dtype=np.float32)
        self.lowbound = np.array(lowbound, dtype=np.float32)


class Electrochemical_LFP:
    def __init__(self, soc, rated_capacity):
        self.soc = soc
        self.rated_capacity = rated_capacity
        
        self.parameters_dict = {
            'Area': {"val": 4.4286, "up": 4.6286, "low": 4.1286},
            
            'L_pos': {"val": 7.3E-05, "up": 8.3E-05, "low": 6.3E-05},
            'epss_pos': {"val": 0.65, "up": 0.75399, "low": 0.55399},
            'L_neg': {"val": 6E-05, "up": 7E-05, "low": 5E-05},
            'epss_neg': {"val": 0.65, "up": 0.75, "low": 0.55},
            'L_sep': {"val": 1.5e-05, "up": 1.5e-04, "low": 1.5e-06},
            'epsl_sep': {"val": 0.50, "up": 0.60, "low": 0.40},
            'csmax_pos': {"val": 22806.0, "up": 25806.0, "low": 20806},
            'csmax_neg': {"val": 33133.0, "up": 34133.0, "low": 32133},
            'rp_pos': {"val": 6.0e-07, "up": 1.4e-06, "low": 1.4e-08},
            'rp_neg': {"val": 1.0e-06, "up": 1.09e-05, "low": 1.09e-08},
            'Ds_pos': {"val": 6.0e-15, "up": 6.327e-13, "low": 1.327e-18},
            'Ds_neg': {"val": 3.0e-15, "up": 3.07e-11, "low": 3.07e-17},
            'epsl_pos': {"val": 0.306008, "up": 0.46008, "low": 0.146008},
            'epsl_neg': {"val": 0.301212, "up": 0.41, "low": 0.141212},
            'ce_0': {"val": 1200, "up": 2258.25410, "low": 558.25410},
            'tc': {"val": 0.2569, "up": 0.669, "low": 0.1569},
            'm_pos': {"val": 6.38e-06, "up": 6.28e-05, "low": 6.18e-8},
            'm_neg': {"val": 5.97e-06, "up": 5.87e-05, "low": 5.27e-8},
            'cs0_pos': {"val": 300, "up": 4806.0, "low": 500},
            'cs0_neg': {"val": 26000, "up": 28062.89, "low": 23000},
            'ks_pos': {"val": 1, "up": 15, "low": 0.50},
            'ks_neg': {"val": 200, "up": 900, "low": 50},
            'resist': {'val': 0.000195, 'up': 0.00099, 'low': 0},
            'e_D_p': {'val': 47000.0, 'up': 235000.0, 'low': -47000.0},
            'e_D_n': {'val': 30300.0, 'up': 60600.0, 'low': -30300.0},
            'e_mref_p': {'val': 39570, 'up': 395700, 'low': -39570},
            'e_mref_n': {'val': 35000, 'up': 350000, 'low': -35000}}

        self.ocv_list = [2.5, 2.7404, 2.9063, 3.0001, 3.067, 3.1183, 3.1584, 3.1879, 3.1994, 3.203, 3.2051, 3.2065,
                         3.2076, 3.2083, 3.2087, 3.2081, 3.2364, 3.2577, 3.275, 3.2842, 3.2862, 3.2869, 3.2874, 3.2881,
                         3.2897, 3.2988, 3.3239, 3.3258, 3.3258, 3.3258, 3.3258, 3.3258, 3.328, 3.3282, 3.3283, 3.3287,
                         3.3317, 3.4435, 3.65]
        self.soc_list = [0.002598336, 0.00331462, 0.005440087, 0.015003394, 0.024566701, 0.034130743, 0.043694054,
                         0.053257659, 0.062821124, 0.072385133, 0.081949159, 0.091513159, 0.101077163, 0.1106412,
                         0.120204677, 0.129768153, 0.139331332, 0.187146484, 0.234961653, 0.282776094, 0.330590406,
                         0.378405576, 0.426219889, 0.47403421, 0.52184938, 0.569664536, 0.617479341, 0.665294445,
                         0.713108749, 0.760923057, 0.808738183, 0.856552601, 0.90436702, 0.952182145, 0.961745323,
                         0.971308777, 0.980872776, 0.990435953, 1]
        
        self.upbound = []
        self.lowbound = []
        self.p0 = []

    def get_up_low_bound(self):
        '''
        param capacity: 额定容量
        param soc: 当前SOC,缺省时按照工况电压计算SOC
        return:p0,upbound,lowbound
        '''
        # 如果不输入SOC，就根据电压计算，如果输入，按照输入来
        if self.soc is None:
            try:
                self.soc = [np.interp(self.vol_high[0], self.ocv_list, self.soc_list)]
            except:
                self.soc = np.array([1])
        if self.soc.max() > 100 or self.soc.min() < 0:
            raise ValueError("Input SOC should be between 0 and 1.")
        elif self.soc.max() > 1:
            # self.soc /= 100
            self.soc = self.soc / 100
        
        # # 01-23 16:10 注释 Area的 更新, 即将上下限固定为 'Area': {"val": 4.4286, "up": 4.6286, "low": 4.1286}
        # # ZT105Ah实测面积4.4286,单位安时面积0.0421,海基120Ah实测面积5.65,单位安时面积:0.0471
        self.parameters_dict['Area']['low'] = self.rated_capacity * 0.042 * 0.8
        self.parameters_dict['Area']['up'] = self.rated_capacity * 0.047 * 1.2
        
        self.parameters_dict['cs0_neg']['low'] = self.parameters_dict['csmax_neg']['low'] * max(self.soc[0] - 0.3, 0.02)
        self.parameters_dict['cs0_neg']['up'] = self.parameters_dict['csmax_neg']['val'] * min(self.soc[0] + 0.2, 0.95)
        # 检查范围是否为负
        self.parameters_dict['cs0_pos']['low'] = self.parameters_dict['csmax_pos']['low'] * max(1 - self.soc[0] - 0.2,
                                                                                                0.02)
        self.parameters_dict['cs0_pos']['up'] = self.parameters_dict['csmax_pos']['up'] * min(1 - self.soc[0] + 0.2,
                                                                                              0.95)

        for key in self.parameters_dict.keys():
            self.upbound.append(self.parameters_dict[key]['up'])
            self.lowbound.append(self.parameters_dict[key]['low'])
            self.p0.append(random.uniform(self.parameters_dict[key]['low'], self.parameters_dict[key]['up']))

        print(f'up : {self.upbound}')
        print(f'low : {self.lowbound}')
        print(f'p0 : {self.p0}')



'''
    参数辨识 输出的 集总参数，顺序固定
'''
def obtain_lumped_param_name_list():
    columns = ['loss', 'current_capacity (Ah)', 'total_capacity (Ah)', 'current_energy (KWh)',
                'discharge_energy (KWh)', 'charge_energy (KWh)', 'mse_loss', 
                'ohmic_ir (mΩ)', 'eta_r_ir (mΩ)', 'connected_component_ratio']  # 添加 "loss" 列标题到最前面

    return columns


'''
    参数辨识 输出的 默认 28 列参数名，顺序固定
'''
def get_param_name_list():
    columns = ["Area","L_pos","epss_pos","L_neg","epss_neg","L_sep","epsl_sep","csmax_pos","csmax_neg",
            "rp_pos","rp_neg","Ds_pos","Ds_neg","epsl_pos","epsl_neg","ce_0","tc","m_pos","m_neg",
            "cs0_pos","cs0_neg","ks_pos","ks_neg","resist","e_D_p","e_D_n","e_mref_p","e_mref_n",
            ] 
    return columns

    
'''
    将参数固定为只取这些列，原因是在 GPU 上进行全方向搜索时，输入参数限制最多只能有这 28 列，且顺序固定
    而且传入 GPU 代码并行计算 损失时必须按照这个固定列
'''
def reset_param_columns_to_fit_vol(df_param):
    # columns = ["Area","L_pos","epss_pos","L_neg","epss_neg","L_sep","epsl_sep","csmax_pos","csmax_neg",
    #         "rp_pos","rp_neg","Ds_pos","Ds_neg","epsl_pos","epsl_neg","ce_0","tc","m_pos","m_neg",
    #         "cs0_pos","cs0_neg","ks_pos","ks_neg","resist","e_D_p","e_D_n","e_mref_p","e_mref_n",
    #         ]  
    columns = get_param_name_list()
    df_param = df_param[columns]

    return df_param


"""
    作用是将多次辨识的结果拼接起来 - 保证辨识结果行数足够
"""
def concat_param_of_different_identify(df_param_this_time, param_iden_this_count):
    
    # df_param_this_time = df_param_this_time._append(param_iden_this_count, ignore_index=True)
    df_param_this_time = df_param_this_time.append(param_iden_this_count, ignore_index=True)    # musa 服务器上的 Python 用这个版本
    print('df_param_this_time', len(df_param_this_time))     
    
    return df_param_this_time

def select_params_based_on_loss(params_inden, select_rows=0):
    rmse_col_name = r'loss'     # 'loss' 列
    
    if select_rows > 0:
        select_param_rows = params_inden[rmse_col_name].values.argsort()[:select_rows]      # 从小到大排列  
    else:
        select_param_rows = params_inden[rmse_col_name].values.argsort()
        
    parameters = params_inden.iloc[select_param_rows, :].reset_index(drop=True)   
    
    return parameters


# """
#     调用参数辨识 : dufv_obj.df_real 的索引时间类型已经是 Float64Index 类型
# """
# def param_identify_then_filter(dufv_obj, sipp_obj, max_iter=500):
    
#     rdi_obj = Real_Data_Info(dufv_obj)
#     rdi_obj.get_real_data_info()
    
#     time_arr = rdi_obj.time_arr
#     current_arr = rdi_obj.current_arr
#     voltage_arr = rdi_obj.voltage_arr
#     temperature_arr = rdi_obj.temperature_arr
#     soc_arr = rdi_obj.soc_arr
#     # 参数辨识输出的 默认 28 列参数
#     param_name_list = get_param_name_list()  
#     lumped_param_name_list = obtain_lumped_param_name_list()
    
#     df_param_all_concat = pd.DataFrame()
#     count = 0
    
#     while df_param_all_concat.shape[0] < sipp_obj.iden_result_len_threshold and count < sipp_obj.each_cell_inden_cnt_Max:
          
#         ep_obj = Elecrochemical_platform(time_arr, current_arr, voltage_arr, temperature_arr, soc_arr, dufv_obj.full_cap, 
#                                         battery_type=sipp_obj.battery_type,
#                                         so_file_str=sipp_obj.so_file_url,
#                                         loss_func=sipp_obj.loss_func_str,
#                                         param_name_list=param_name_list, lumped_param_name_list=lumped_param_name_list)
                       
#         ep_obj.ep_obj_run(particle_num=64, para_num=28, max_iter=max_iter, task_num=sipp_obj.iden_task_num_Max, all_gather=1)           
                             
#         params_inden, params_up_low = ep_obj.parameters_iden, ep_obj.params_up_low_bound_dict

#         df_param_all_concat = concat_param_of_different_identify(df_param_all_concat, params_inden)
#         count += 1
#         print(f'第 {dufv_obj.vol_name} 节电池 辨识了 {count} 次')
        
#     # params_inden = params_inden[dufv_obj.vol_name].reset_index(drop=True) 
#     params_inden = df_param_all_concat.copy().reset_index(drop=True) 
#     params_up_low = pd.DataFrame(params_up_low).loc[['up', 'low']].copy()
#     print(f'params_inden: {params_inden}')
#     print(f'params_up_low: {params_up_low}')
    
#     # select_rows=0 表示 只将 参数 根据 loss 排序，不筛选
#     parameters = select_params_based_on_loss(params_inden, select_rows=0)
#     # # 表示 根据 loss 排序，并筛选参数
#     # parameters = select_params_based_on_loss(params_inden, select_rows=sipp_obj.select_rows_from_param_iden)
    
#     print(f"[After Select] parameters: {parameters[['loss', 'Area', 'L_pos']]}")
#     # assert 0
    
#     return parameters, params_up_low


"""
    电化学类 Elecrochemical_platform 及 其继承类 的实例化函数，返回类的实例
"""
def create_electro_obj(dufv_obj, sipp_obj, input_class, **kwargs):
    rdi_obj = Real_Data_Info(dufv_obj)
    rdi_obj.get_real_data_info()
    
    time_arr = rdi_obj.time_arr
    current_arr = rdi_obj.current_arr
    voltage_arr = rdi_obj.voltage_arr
    temperature_arr = rdi_obj.temperature_arr
    soc_arr = rdi_obj.soc_arr
    
    # for i in [time_arr, current_arr, voltage_arr, temperature_arr, soc_arr]:
    #     print(i)
    # assert 0
    
    # 参数辨识输出的 默认 28 列参数
    param_name_list = get_param_name_list()  
    # 集总参数
    lumped_param_name_list = obtain_lumped_param_name_list()

    # # 使用 *args 和 **kwargs 来传递额外的参数给 input_class 的构造函数  
    # electro_obj = input_class(time_arr, current_arr, voltage_arr, temperature_arr, soc_arr, dufv_obj.full_cap, 
    #                         battery_type=sipp_obj.battery_type,
    #                         so_file_str=sipp_obj.so_file_url,
    #                         loss_func=sipp_obj.loss_func_str,
    #                         param_name_list=param_name_list, lumped_param_name_list=lumped_param_name_list,
    #                         *args, **kwargs) 
    print(f'kwargs : {kwargs}')
    # 合并额外的参数，均采用关键字方式调用
    class_init_argument_dict = {  
        'time': time_arr,  
        'current': current_arr,  
        'voltage': voltage_arr,  # 从某个地方获取  
        'temperature': temperature_arr,  
        'soc': soc_arr,  
        'rated_capacity': dufv_obj.full_cap,  
        'battery_type': sipp_obj.battery_type,  
        'so_file_str': sipp_obj.so_file_url,  
        'loss_func': sipp_obj.loss_func_str,  
        'param_name_list': param_name_list,  
        'lumped_param_name_list': lumped_param_name_list,  
        # 如果 kwargs 包含额外的关键字参数，则合并它们
        # **(kwargs or {})  
    }
    
    if kwargs:  # 参数不为空
        class_init_argument_dict.update(kwargs)
        # print(class_init_argument_dict)
        # print(class_init_argument_dict.keys())
            
    electro_obj = input_class(**class_init_argument_dict)

    return electro_obj


"""
    调用参数辨识 : dufv_obj.df_real 的索引时间类型已经是 Float64Index 类型 【为了通用，替换上面注释的代码】
"""
def param_identify_then_filter(dufv_obj, sipp_obj, max_iter=500):
    
    # rdi_obj = Real_Data_Info(dufv_obj)
    # rdi_obj.get_real_data_info()
    
    # time_arr = rdi_obj.time_arr
    # current_arr = rdi_obj.current_arr
    # voltage_arr = rdi_obj.voltage_arr
    # temperature_arr = rdi_obj.temperature_arr
    # soc_arr = rdi_obj.soc_arr
    # # 参数辨识输出的 默认 28 列参数
    # param_name_list = get_param_name_list()  
    # lumped_param_name_list = obtain_lumped_param_name_list()
    
    df_param_all_concat = pd.DataFrame()
    count = 0
    
    while df_param_all_concat.shape[0] < sipp_obj.iden_result_len_threshold and count < sipp_obj.each_cell_inden_cnt_Max:
          
        # ep_obj = Elecrochemical_platform(time_arr, current_arr, voltage_arr, temperature_arr, soc_arr, dufv_obj.full_cap, 
        #                                 battery_type=sipp_obj.battery_type,
        #                                 so_file_str=sipp_obj.so_file_url,
        #                                 loss_func=sipp_obj.loss_func_str,
        #                                 param_name_list=param_name_list, lumped_param_name_list=lumped_param_name_list)
        
        ep_obj = create_electro_obj(dufv_obj, sipp_obj, Elecrochemical_platform)
        
        print(f'ep_obj.time:{ep_obj.time}')
        print(f'ep_obj.current:{ep_obj.current}')
        print(f'ep_obj.voltage:{ep_obj.voltage}')
        print(f'ep_obj.temperature:{ep_obj.temperature}')
        print(f'ep_obj.soc:{ep_obj.soc}')
        print(f'ep_obj.battery_type:{ep_obj.battery_type}')
        print(f'ep_obj.so_file_str:{ep_obj.so_file_str}')
        print(f'ep_obj.rated_capacity:{ep_obj.rated_capacity}')
        
        ep_obj.ep_obj_run(particle_num=64, para_num=28, max_iter=max_iter, task_num=sipp_obj.iden_task_num_Max, all_gather=1)           
                             
        params_inden, params_up_low = ep_obj.parameters_iden, ep_obj.params_up_low_bound_dict

        df_param_all_concat = concat_param_of_different_identify(df_param_all_concat, params_inden)
        count += 1
        print(f'第 {dufv_obj.vol_name} 节电池 辨识了 {count} 次')
        
    # params_inden = params_inden[dufv_obj.vol_name].reset_index(drop=True) 
    params_inden = df_param_all_concat.copy().reset_index(drop=True) 
    params_up_low = pd.DataFrame(params_up_low).loc[['up', 'low']].copy()
    print(f'params_inden: {params_inden}')
    print(f'params_up_low: {params_up_low}')
    
    # select_rows=0 表示 只将 参数 根据 loss 排序，不筛选
    parameters = select_params_based_on_loss(params_inden, select_rows=0)
    # # 表示 根据 loss 排序，并筛选参数
    # parameters = select_params_based_on_loss(params_inden, select_rows=sipp_obj.select_rows_from_param_iden)
    
    print(f"[After Select] parameters: {parameters[['loss', 'Area', 'L_pos']]}")
    # assert 0
    
    return parameters, params_up_low


