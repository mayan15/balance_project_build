"""
    注意：本文件 包含 全方向搜索 ods - Python / cuda 版
"""
import numpy as np
import pandas as pd

import itertools 
import ctypes  
import time

"""
    log.logger.debug("用于打印日志")
"""
from tool_utils.pvlog import Logger, level_SOH_electro
log = Logger('./logs/SOH_electro.log', level=level_SOH_electro)

"""
    storage_database : 所需数据类
"""
from .storage_database import Volt_Fit_and_Loss, Omni_Directions_Several_Steps, Gradient_Result_Record

"""
    sift_param_via_distance : 确定每行 选择参数 的 搜索范围
"""
from .sift_param_via_distance import convert_Series_to_DataFrame

"""
    call_electrochem_identify : 参数辨识相关的接口，也包含参数的列名获取，不同行 DataFrame
"""
from .call_electrochem_identify import Elecrochemical_platform, create_electro_obj, reset_param_columns_to_fit_vol, concat_param_of_different_identify

"""
    electrochem_mpi : mpi 调用 .so 文件的接口
"""
from .electrochem_mpi import MpiEC


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


class Omni_Direction_Try_Steps(object):
    
    """
        初始化参数：
        change_columns - 变化且需要搜索的列名
        param_this_row - 当前输入的 1行 参数 pd.DataFrame
        params_up_low - 当前参数的上限 & 下限 pd.DataFrame
        batch_size - 批量生成多元素生成器时 单批次个数
    """
    def __init__(self, change_columns, param_this_row, params_up_low, batch_size):
        self.change_columns = change_columns
        self.param_this_row = param_this_row
        self.params_up_low = params_up_low
        self.batch_size = batch_size
    
    """
        输入：change_columns 当前元素 想对其上下限的差值
        输出：基于输入，输出所有的组合，比如 list1有3个元素，则输出个数为 2^3=8个 含3个元素的 generator
    """
    def all_combinations_generator(self, list1, list2):  
        for combo in itertools.product([0, 1], repeat=len(list1)):  
            yield [list1[i] if bit == 0 else list2[i] for i, bit in enumerate(combo)]  
    
    def get_step_cnts_based_on_change_cols(self, change_param_num):  
        """
            thresholds 第 1 个值表示 change_param_num 的个数，第 2 个值表示 step 的个数
            则下面的例子为：<=5 则 step=50000; <=10 则 step=5000; <=15 则 step=1000; <=19 则 step=500; >19 则 step=100
        """
        # 定义阈值和对应的cnts值   
        thresholds = [(5, 50000), (10, 5000), (15, 1000), (19, 500)] 
        default_cnts = 100  # 默认值，当change_param_num大于所有阈值时使用
        
        # 遍历阈值列表，找到第一个大于或等于change_param_num的阈值，并返回对应的cnts  
        for threshold, cnts in thresholds:  
            if change_param_num <= threshold:  
                return cnts  

        log.logger.debug(f'cnts:{cnts}\t')
        # print('cnts:\t', cnts)
        # 如果没有找到匹配的阈值（即change_param_num大于所有阈值），则返回默认值  
        return default_cnts  
        
    """
        输入：步长个数 
        输出：输出含 cnts 个元素的 ndarray, 比如 cnts=5 可以输出 array([0.88592191, 0.36308118, 0.58177421])
        注意：这里的 start_val 和 end_val 必须在 0~1 之间
    """
    def get_multi_factor(self, cnts=5):  
        # start_val = 0.1  
        # end_val = 0.9
        start_val = 1e-2 
        end_val = 0.99   
        multi_factor = np.random.uniform(start_val, end_val, cnts)
        if 1 not in multi_factor:
            # 使用索引创建一个新的数组，包含新元素和原始数组的元素  
            multi_factor = np.insert(multi_factor, 0, 1)  
            multi_factor = np.sort(multi_factor)
        return multi_factor
        # return np.random.uniform(start_val, end_val, cnts)    # 不加 数值 1，直接返回
    
    """
        输入：变化参数的个数 - change_param_num = len(change_columns)
        输出：含 cnts 行 change_param_num 列 ndarray 的 array
    """
    def get_step_iters_on_multipliers(self, change_param_num):  
        # 这里我们返回一个列表，其中包含 change_param_num 个步长数组  
        cnts = self.get_step_cnts_based_on_change_cols(change_param_num)
        # return np.array([self.get_multi_factor(cnts) for _ in range(change_param_num)]).T  # 要记得转置，每个 参数列 用的随机数是 不同 的
        return np.tile(self.get_multi_factor(cnts), (change_param_num,1)).T    # 这样每个 参数列 用的随机数是 一样 的
    
    """
        输入：
            combinations 是基于上下边界生成的 组合，共 2^change_param_num 个元素
            steps_iter 是生成的
        输出：含 cnts 行 change_param_num 列 ndarray 的 array
    """
    def generate_multiplied_results(self, combinations, steps_iter):  
        for combination in combinations:  
            combination_array = np.array(combination)  
            for step_array in steps_iter:  
                # result_array = combination_array.copy()  # 复制以避免原地修改  
                result_array = combination_array * step_array  # 只对指定的列应用步长  
                yield result_array  
  
    def generate_params_array_group(self, combinations, steps_iter):  
        results_iter = self.generate_multiplied_results(combinations, steps_iter)  
        batch = []  
        for result_array in results_iter:  
            batch.append(result_array)  
            if len(batch) == self.batch_size:  
                yield batch  
                batch = []  
        if batch:  
            yield batch
        
    def get_max_change_threshold_of_each_param(self):
        
        params_up = self.params_up_low.loc['up'].to_dict()
        params_low = self.params_up_low.loc['low'].to_dict()

        param_this_row_arr_1d = self.param_this_row[self.change_columns].values.flatten()    # 一维
        
        up = [params_up[k] for k in self.change_columns if k in params_up]
        low = [params_low[k] for k in self.change_columns if k in params_low]
        
        # 使用ndarray计算差值 - 注意：都是以 param_this_row 去减去别的的值
        self.difference_up_value = param_this_row_arr_1d - np.asarray(up)
        self.difference_value_low = param_this_row_arr_1d - np.asarray(low)
        # 以 difference_up_value & difference_value_low 为 基准步长去生成 all_combinations_generator

        # print('param_this_row_arr_1d', param_this_row_arr_1d)
        # # print('params_up', params_up)
        # print('up', up)
        # # print('params_low', params_low)
        # print('low', low)
        # print('self.difference_up_value', self.difference_up_value)
        # print('self.difference_value_low', self.difference_value_low)
        
    """
        下面的方法是应用在 多参数组合在 musa gpu上生成时的方法
    """
    def get_index_of_change_columns_before_adopting_gpu(self):    
        # 注意参数的 列顺序 必须对
        log.logger.debug(f'self.params_up_low : {self.params_up_low.columns}')
        # print(f'self.params_up_low : {self.params_up_low.columns}')
        param0_1d = self.param_this_row.values.flatten()    # 一维
        log.logger.debug(f'all_columns_of_param_this_row : {list(self.param_this_row.columns)}')
        # print(f'all_columns_of_param_this_row : {list(self.param_this_row.columns)}')
        all_columns_of_param_this_row = list(self.param_this_row.columns)
        # 参数的上下界 的列 与 param_this_row 列一样
        up = self.params_up_low.loc['up'][all_columns_of_param_this_row].values.flatten()    # 一维
        low = self.params_up_low.loc['low'][all_columns_of_param_this_row].values.flatten()    # 一维
        
        # 使用ndarray计算差值 - 注意：都是以 param_this_row 去减去别的的值
        self.dpara_up = param0_1d - np.asarray(up)
        self.dpara_down = param0_1d - np.asarray(low)
        
        log.logger.debug(f'param0_1d:{param0_1d}')
        log.logger.debug(f'up:{up}')
        log.logger.debug(f'low:{low}')
        log.logger.debug(f'dpara_up:{self.dpara_up}')
        log.logger.debug(f'dpara_down:{self.dpara_down}')
        # print(f'param0_1d:{param0_1d}')
        # print(f'up:{up}')
        # print(f'low:{low}')
        # print(f'dpara_up:{self.dpara_up}')
        # print(f'dpara_down:{self.dpara_down}')
        
        # 待计算参数位置1，计算总计待变更参数个数
        self.flags_target = int(0)
        self.num_target_para = 0
        
        for column_name in self.change_columns:
            index = all_columns_of_param_this_row.index(column_name)      # self.param_this_row 的列顺序
            self.flags_target = self.flags_target | (1 << index)
            self.num_target_para += 1

        # 参数集转为numpy
        self.para_target = np.array(param0_1d)
    
    def allot_steps_on_gpu(self, change_param_num):    
        """
            thresholds 第 1 个值表示 change_param_num 的个数，第 2 个值表示 step 的个数
            则下面的例子为：<=5 则 step=50000(仅举例)
        """
        # 定义阈值和对应的cnts值   
        thresholds = [(2, int(1e8)), (5, int(1e7)), (10, int(1e5)), (15, int(1e4)), (19, int(1e3))] 
        # 当变化 24 个参数时，pow(2,24)*200 ~ 接近34亿
        default_cnts = 200  # 默认值，当change_param_num大于所有阈值时使用
        
        # 遍历阈值列表，找到第一个大于或等于change_param_num的阈值，并返回对应的cnts  
        for threshold, cnts in thresholds:  
            if change_param_num <= threshold:  
                return cnts  

        log.logger.debug(f'cnts:{cnts}\t')
        # print('cnts:\t', cnts)
        # 如果没有找到匹配的阈值（即change_param_num大于所有阈值），则返回默认值  
        return default_cnts  


"""
    输入：一维 array
    输出：True / False / None
    备注：该函数在功能上不是必须的，是为了检测 参数-上限/下限 是否同号（应当同号，即全正或者全负），不同号说明参数的边界区间出错
"""
def are_same_sign_except_zero(arr, first_element_cnts):  
    # # 移除0值  
    # non_zero_arr = arr[arr != 0]  
    # 取前 _ 个值 - 前 change_columns 个参数
    non_zero_arr = arr[:first_element_cnts]
      
    # 如果数组中没有非零元素，则无法判断符号  
    if non_zero_arr.size == 0:  
        return None  # 或者返回True/False，取决于你的具体需求  
      
    # 获取第一个非零元素的符号  
    first_sign = np.sign(non_zero_arr[0])  
      
    # 检查所有非零元素是否都与第一个非零元素同号  
    return np.all(np.sign(non_zero_arr) == first_sign)  


"""
    类输入参数：
        同 Elecrochemical_platform
        parameters - pd.DataFrame 【注意：在调用 filter_params_by_ods 时只能输入一行参数，因为是在该行参数周边进行搜索】
        soh_col_name 列名 - str 【不一定用得上】
"""
class Elecrochemical_platform_with_Parameters(Elecrochemical_platform):
    
    def __init__(self, time, current, voltage, temperature, soc, rated_capacity, 
                 battery_type, so_file_str, loss_func,
                 param_name_list, lumped_param_name_list,
                 parameters, soh_col_name='SOH'):  
        
        # 注意这里不再传递 self，并且只传递应该传递给基类的参数  
        super().__init__(time, current, voltage, temperature, soc, rated_capacity, 
                        battery_type, so_file_str, loss_func,
                        param_name_list, lumped_param_name_list)
        
        self.parameters = parameters
        self.soh_col_name = soh_col_name
    
    def preprocess_params(self, ):
        if not isinstance(self.parameters, (np.ndarray)):
            para_target = self.parameters.values.astype(np.float32).ravel()       # 视图
        else:
            para_target = self.parameters
            
        return para_target
    
    def align_pointer_type(self, para_target, dpara_up, dpara_down):
        para_target = np.array(para_target, dtype=np.float32)
        dpara_up = np.array(dpara_up, dtype=np.float32)
        dpara_down = np.array(dpara_down, dtype=np.float32)
        
        return para_target, dpara_up, dpara_down
        
    # 参数 all_gather = 1  # 是否每个进程收集全部结果，1为是 
    def filter_params_by_ods(self, dpara_up, dpara_down, flags_target, comb_multiple, para_num, target_num, all_gather=1):  
        
        # 预处理数据 - 分电池类型去处理
        self.pre_process_data()
        # 获取边界和初始参数 - 分电池类型去处理 【即：更新 电池类型 的截止电压】
        self.update_param_bound(get_bound_flag=False)    # 不用更新参数边界  
        
        para_target = self.preprocess_params()
        
        start_time = time.time()
        log.logger.debug('start [filter_params_by_ods] !!')
        # print('start [filter_params_by_ods] !!')

        # 实例化，获取结果
        solver = MpiEC(all_gather)
        solver.data_init(self.so_file_str, self.time, self.current, self.voltage, self.temperature, para_num)
        
        # print(f'solver.so_lib: {solver.so_lib}')
        # print(f"self.so_file_str : {self.so_file_str}")
        # print(f"self.time : {self.time}")
        # print(f"self.current : {self.current}")
        # print(f"self.voltage : {self.voltage}")
        # print(f"self.temperature : {self.temperature}")
        # print(f"para_num : {para_num}")
        
        # print(f"self.rated_capacity : {self.rated_capacity}")
        # print(f"self.cut_voltage_up_limit : {self.cut_voltage_up_limit}")
        # print(f"self.cut_voltage_low_limit : {self.cut_voltage_low_limit}")
        # print(f"self.loss_func : {self.loss_func}")
        # print(f"self.battery_type : {self.battery_type}")
        
        para_target, dpara_up, dpara_down = self.align_pointer_type(para_target, dpara_up, dpara_down)
        
        # 结果计算
        arr_2d = solver.get_ods_result(para_target, dpara_up, dpara_down, comb_multiple, target_num, flags_target, 
                                       capacity_real=self.rated_capacity,
                                       cut_voltage_up=self.cut_voltage_up_limit, cut_voltage_low=self.cut_voltage_low_limit,
                                       loss_func=self.loss_func, battery_type=self.battery_type)

        # print(f"arr_2d : {arr_2d}")
        
        min_index = np.argmin(arr_2d[:,0]) 
        min_loss = arr_2d[:,0][min_index]
        log.logger.debug(f'(python) rank:{solver.rank}  min_index: {min_index}  max_time:{np.max(solver.all_time)}  min_loss: {min_loss}')
        # print(f'(python) rank:{solver.rank}  min_index: {min_index}  max_time:{np.max(solver.all_time)}  min_loss: {min_loss}')
        
        # 备注：这里参数直接是生成 DataFrame输出的，要注意 参数的列 与 all_cols 是否一致
        df = self.post_process_parameters(arr_2d, param_filter_flag=False)
        # log.logger.debug(f'df: {df}')
        log.logger.debug(f'df.columns: {df.columns}')
        # log.logger.debug(f"df['loss']: {df['loss']}")
        log.logger.debug(f"df['loss']: {df[['loss', 'Area']]}")
        log.logger.debug(f"df['total_capacity (Ah)']: {df[['total_capacity (Ah)', 'SOH']]}")
        log.logger.debug(f"self.rated_capacity : {self.rated_capacity}")
        # print(f'df: {df}')
        # print(f'df.columns: {df.columns}')
        # print(f"df['loss']: {df['loss']}")
        # print(f"df['loss']: {df[['loss', 'Area']]}")
        # print(f"df['total_capacity (Ah)']: {df[['total_capacity (Ah)', 'SOH']]}")
        # print(f"self.rated_capacity : {self.rated_capacity}")
        # # # 数据保存
        # # column_names = lumped_para_names + para_name_list
        # df = pd.DataFrame(arr_2d, columns=self.column_names)  # 将 numpy 数组转换为 pandas DataFrame，并设置列标题
        # SOH 计算后传给 self.parameters 的 'SOH' 列
#         self.parameters[self.soh_col_name] = capacity / self.dufv_obj.full_cap

        end_time = time.time()
        get_loss_time = end_time - start_time
        log.logger.debug(f'[filter_params_by_ods] : {get_loss_time}')
        # print(f'[filter_params_by_ods] : {get_loss_time}')
        
        return df
    
    def test(self, para_num, task_num=1000, all_gather=1):
        
        # 预处理数据 - 分电池类型去处理
        self.pre_process_data()
        # 获取边界和初始参数 - 分电池类型去处理 【即：更新 电池类型 的截止电压】
        self.update_param_bound(get_bound_flag=False)    # 不用更新参数边界  
        
        para_target = self.preprocess_params()
        
        # start_time = time.time()
        # print('start [filter_params_by_ods] !!')

        # 实例化，获取结果
        solver = MpiEC(all_gather)
        solver.data_init(self.so_file_str, self.time, self.current, self.voltage, self.temperature, para_num)
        
        # para_target = np.array(np.tile(para_target, (task_num, 1))).flatten()
        para_target = para_target.flatten()
        
        # # 测试 get_loss
        # arr_2d = solver.get_loss(para_target, task_num, para_num, self.loss_func)
        # print(f"get_loss arr_2d : {arr_2d}")
        
        # 测试 get_voltage
        arr_2d = solver.get_voltage(para_target, task_num, para_num, self.loss_func, self.battery_type)
        # print(f"get_voltage arr_2d : {arr_2d}")
        
        loss = arr_2d[:, 0]
        voltage = arr_2d[:, 2:]
        log.logger.debug(f"voltage : {voltage}")
        # print(f"voltage : {voltage}")
        
        volt_error = (voltage - self.voltage) * 1000
        log.logger.debug(f"volt_error : {volt_error}")
        # print(f"volt_error : {volt_error}")
        
        
        # # 测试 get_capacity
        # arr_2d = solver.get_capacity(para_target, task_num, para_num, capacity_real=self.rated_capacity,
        #                                cut_voltage_up=self.cut_voltage_up_limit, cut_voltage_low=self.cut_voltage_low_limit,
        #                                loss_func=self.loss_func, battery_type=self.battery_type)
        
        # print(f"get_capacity arr_2d : {arr_2d}")
        
        # assert 0
        
        return np.vstack((voltage, self.voltage)), loss   # 最后一行是真实电压
    


"""
    GPU 版 - 单步迭代 全方向多步长搜索 : cuda 或 musa(GPU生成多组合参数, 直接计算loss并输出)
"""
def update_params_single_step_GPU(Output_data, odss_obj, dufv_obj, sipp_obj):
                                #   so_path, battery_type, loss_type, rmse_min_last_iter):

    # 本次输入的参数 - 全部列 存储
    params_of_all_rows_this_iter_input = Output_data.out_param_df.copy()
    all_cols = params_of_all_rows_this_iter_input.columns
    log.logger.debug(f'all_cols : {all_cols}')     # 确保参数的顺序对
    # print(f'all_cols : {all_cols}')     # 确保参数的顺序对
    assert [col in all_cols for col in odss_obj.change_columns], "Output_data.out_param_df.columns not in change_columns"

    param_after_iter_of_each_param_row = pd.DataFrame()     # 去一行行获取每个参数的迭代结果
    # param_after_iter_of_each_param_row = []   # 列表，每个元素是 Series / ndarray
    # loss_of_each_param_row = []      # 列表，每个元素是 float

    # 输入的参数逐行 输入，一行参数这样去处理
    for i, params_this_iter_input in params_of_all_rows_this_iter_input.iterrows(): 
        # param_one_row 为 DataFrame 类型
        param_one_row = convert_Series_to_DataFrame(params_this_iter_input, chosen_index=i)    # 这里输入的 params_this_iter_input 是 Series，实际上 chosen_index 不起作用
        # 【重要】 参数赋值 时需要注意 选取这些列 - cuda 指定了最多只能有这些列 【保存时保存全部列，进行搜索是只需要输入 所需的 28 列
        param_one_row = reset_param_columns_to_fit_vol(param_one_row)   # 将列顺序转换为 musa/cuda 要求的顺序，这一步实际上可以不用（因为在参数最初传进来时已经固定了顺序），这里只是加一层保险
        
        odss_obj.params_up_low = odss_obj.params_up_low_list[i]     # 该行参数对应的搜索边界
        log.logger.debug(f'odss_obj.params_up_low : {odss_obj.params_up_low}')
        # print(f'odss_obj.params_up_low : {odss_obj.params_up_low}')
        
        # 初始化对象
        odts_obj = Omni_Direction_Try_Steps(odss_obj.change_columns, param_one_row, odss_obj.params_up_low, odss_obj.batch_size)
        
        # 生成 含每个参数的上下限的 ndarray
        odts_obj.get_index_of_change_columns_before_adopting_gpu()      # 在到 gpu 生成多组合参数之前，先确定最终要变换参数列的索引，方便在 gpu 中进行二进制组合
        
        sign_of_dpara_up = are_same_sign_except_zero(odts_obj.dpara_up, len(odss_obj.change_columns))
        log.logger.debug(f"sign_of_dpara_up : {sign_of_dpara_up}")
        # print(f"sign_of_dpara_up : {sign_of_dpara_up}")
        assert sign_of_dpara_up == True, "param_up_low Goes Wrong!"
        
        # 参数初始化
        para_target = odts_obj.para_target.astype(np.float32)
        dpara_up = odts_obj.dpara_up.astype(np.float32)
        dpara_down = odts_obj.dpara_down.astype(np.float32)
        flags_target = odts_obj.flags_target  # 待变更参数二进制位选择
        num_target_para = odts_obj.num_target_para      # len(odss_obj.change_columns)，即需要计算的变化参数的个数

        log.logger.debug(f"odss_obj.change_columns : {odss_obj.change_columns} \n {len(odss_obj.change_columns)}")
        log.logger.debug(f"num_target_para : {num_target_para}")
        # print(f"odss_obj.change_columns : {odss_obj.change_columns} \n {len(odss_obj.change_columns)}")
        # print(f"num_target_para : {num_target_para}")
        # assert 0
        
        comb_multiple = odts_obj.allot_steps_on_gpu(num_target_para)    # 全方向扩展倍数
        # comb_multiple = int(20 * 1)  # 全方向扩展倍数 - 测试时用（输入小值测试）
        
        log.logger.debug(f'comb_multiple : {comb_multiple}')
        # print(f'comb_multiple : {comb_multiple}')
        # comb_num = pow(2, num_target_para)
        para_num = len(para_target)
        # target_num = 1  # 选 几行/几套 参数
        target_num = len(param_one_row)  # 选 几行/几套 参数 - 要求输入为 1行 参数

        log.logger.debug(f'para_target : {para_target}', type(para_target), para_target[0])  
        # print(f'para_target : {para_target}', type(para_target), para_target[0])  
        
        start_time = time.time()

        # # extra_args = (para_target,)  # 注意这是一个元组，对于 *args 来说  
        # # epwp_obj = create_electro_obj(dufv_obj, sipp_obj, Elecrochemical_platform_with_Parameters, para_target)
        # extra_kwargs = {'parameters': para_target, 'soh_col_name':'SOH'}  # 这是一个字典，对于 **kwargs 来说
        # 类名后的参数 必须通过关键字传递
        epwp_obj = create_electro_obj(dufv_obj, sipp_obj, Elecrochemical_platform_with_Parameters, parameters=para_target, soh_col_name='SOH')
        
        # # 测试 - 临时
        # epwp_obj.test(para_num, task_num=1000, all_gather=1)
        
        # mpi 架构 GPU 调用方式 - loss 升序 选择 target_num 组参数 及 对应损失
        param_after_ods = epwp_obj.filter_params_by_ods(dpara_up, dpara_down, flags_target, comb_multiple, para_num, target_num, all_gather=1)
        
        assert [col in all_cols for col in param_after_ods.columns] and len(param_after_ods.columns) == len(all_cols), "Output_data.out_param_df.columns != all_cols"
        
        # 注意 不同 Python 版本用的 append 方法有下划线
        param_after_iter_of_each_param_row = param_after_iter_of_each_param_row._append(param_after_ods, ignore_index=True)
        #param_after_iter_of_each_param_row = param_after_iter_of_each_param_row.append(param_after_ods, ignore_index=True)
        
        end_time = time.time()
        get_loss_time = end_time - start_time

        log.logger.debug(f"== 第 {i} 行参数 == get_loss_time == : {get_loss_time}\n") 
        # print(f"== 第 {i} 行参数 == get_loss_time == : {get_loss_time}\n") 
      
    # 全部行的参数都更新完毕
    Output_data.out_param_df = param_after_iter_of_each_param_row[all_cols].reset_index(drop=True)
    Output_data.out_rmse_list = Output_data.out_param_df['loss'].tolist() 
    
    log.logger.debug(f"== Output_data == after:{Output_data}\n") 
    # print("== Output_data == after:\n", Output_data) 
    
    return Output_data


# -----------------------------------------------------------------------------
# 以下为 
# 外层循环是 迭代次数，内层循环是 每行参数 - 为了最后 grr_obj 输出好保存

"""
基于 输入参数 进行 全方向搜索 调用 的主函数
外层循环是 迭代次数，内层循环是 每行参数 - 这么实现是为了最后 grr_obj 输出好保存
参数：
    parameters - 含所有列名的参数（包括 loss, soh 这些） 【注意】
    change_columns - 需要搜索的参数列名 （通常为24个）
    
    params_up_low_list - parameters 中每行参数对应的 每个参数的搜索边界
    
    dufv_obj - 原始数据保存的类
    sipp_obj - 含辨识需要的输入类
    
"""
def ods_search_params(parameters, change_columns, params_up_low_list, dufv_obj, sipp_obj):
    
    """
    # [step] - 含参数边界 & 迭代信息 的数据类 初始化
    """
    odss_obj = Omni_Directions_Several_Steps(
        params_up_low = pd.DataFrame(),   # 【重要】 这里有多组参数的上下边界，其值是不同的，初始化先置为空
        params_up_low_list = params_up_low_list.copy(),   # 【重要】 这里有多组参数的上下边界，其值是不同的
        change_columns = change_columns,
        batch_size = int(1e6),              # GPU每次并行计算loss的参数组数
        max_iter=1,                  # 允许最大迭代次数
        iterative_cutoff_rmse_critical_value=5e-1)      # 迭代停止的电压拟合误差比较基准值

    # 断定 change_columns 元素全在 parameters.columns 里 
    assert all(elem in set(parameters.columns) for elem in odss_obj.change_columns), "Error! change_columns not all in parameters.columns"
    
    """
    # 初始化 Output_data
      注意： out_param_df 赋值的 parameters 包含全部的列
            out_rmse_list 是取 含所有列参数 parameters 的 ['loss'] 列 再转为 列表
    """
    Output_data = Volt_Fit_and_Loss(vol_out_df = pd.DataFrame(),
                                    out_param_df = parameters.copy(),    # 【重要】 注意这里赋值的参数是 全部列，后面需要处理 - cuda 指定了最多只能有这些列
                                    mse_df = pd.DataFrame(),
                                    out_rmse_list = parameters['loss'].tolist())    # 将 parameters 对应的 loss 直接赋值
    
    """
    # 初始化 grr_obj - 存每次迭代结果
    """
    grr_obj = Gradient_Result_Record(all_params_out_df = pd.DataFrame(),
                                    iter_cnt = [],
                                    param_df_record = [],
                                    rmse_record = [],
                                    param_at_min_rmse = [])
    
    # 多行参数输入 - 进行 全方向搜索
    grr_obj, Output_data = step_optimizer_by_loss(Output_data, odss_obj, dufv_obj, grr_obj, sipp_obj)                     
                                                #   sipp_obj.so_file_url, sipp_obj.battery_type, sipp_obj.loss_func_str)

    # grr_obj 还要选出每个参数对应的 loss最小的结果                                        
    grr_obj = process_gradient_output(grr_obj, odss_obj.change_columns)
    log.logger.debug(f'== Final Output == grr_obj:\n {grr_obj}')
    # print(f'== Final Output == grr_obj:\n {grr_obj}')
    
    return grr_obj
    
"""
    这里是 多行参数 的 多步迭代 和更新，输入：
    Output_data - 当前步骤的参数 (DataFrame) + 参数对应的 loss (list, DataFrame)
    odss_obj - 含参数边界 & 迭代信息 的数据类
    dufv_obj - 真实电压数据类
    grr_obj - 包含每步迭代的参数保存 数据类 - 用于保存迭代结果的
    
    --- 以下三个变量调用来自 sipp_obj
    so_path - .so 文件路径，电压拟合计算 loss 用
    battery_type - 电池类型
    loss_type - 损失类型
"""
def step_optimizer_by_loss(Output_data, odss_obj, dufv_obj, grr_obj, sipp_obj):
                        #    so_path, battery_type, loss_type):
    
    change_columns = odss_obj.change_columns

    # 外层循环是迭代，进行优化迭代 - 注意 Output_data.out_param_df 是多行 DataFrame
    for t in range(odss_obj.max_iter):      
        # 参数保存 - 初始参数
        if t == 0:  # 初始数据存储:   
            # 为了保存初始参数，这里需要将 Output_data.out_param_df 先存成 
            log.logger.debug(f'Output_data : {Output_data}')
            # print(f'Output_data : {Output_data}')
            # Output_data.out_rmse_list = Output_data.out_param_df['loss']    # 将所选参数 辨识的 loss 赋值给 Output_data.out_rmse_list
            grr_obj = record_gradient_variable_over_iteration(Output_data, change_columns, grr_obj, t)
            log.logger.debug(f'Output_data.out_rmse_list : {Output_data.out_rmse_list}')
            # print(f'Output_data.out_rmse_list : {Output_data.out_rmse_list}')
        
        t += 1  
        # 从 t = 1 计算结束后，收集每行参数的 loss 值 
        if t > 1:
            rmse_min = rmse_min_last_iter
        else:
            rmse_min = Output_data.out_rmse_list
            
        log.logger.debug(f't: {t}, \t rmse_min: {rmse_min}')      # 注意：在实际打印输出时，最后一次迭代的结果不会打印，因为程序终止了，也就是说只能打印到倒数第二次的迭代结果
        # print(f't: {t}, \t rmse_min: {rmse_min}')      # 注意：在实际打印输出时，最后一次迭代的结果不会打印，因为程序终止了，也就是说只能打印到倒数第二次的迭代结果
        
        # # 不在这里处理，因为想要取全部的列名
        # # 【重要】 参数赋值 时需要注意 选取这些列 - cuda 指定了最多只能有这些列 【保存时保存全部列，进行搜索是只需要输入 所需的 28 列
        # Output_data.out_param_df = reset_param_columns_to_fit_vol(Output_data.out_param_df.copy())
        
        # 参数 全方向全步长搜索
        log.logger.debug(f'开始 全方向 搜索 - 输入 GPU 进行搜索')
        # print(f'开始 全方向 搜索 - 输入 GPU 进行搜索')
        Output_data = update_params_single_step_GPU(Output_data, odss_obj, dufv_obj, sipp_obj) 
        # Output_data = update_params_single_step_GPU(Output_data, odss_obj, dufv_obj, so_path, battery_type, loss_type, rmse_min)     # rmse_min - list 类型，但是子函数内部没用上
        
        # 参数保存【全列参数】 - 包含初始参数
        grr_obj = record_gradient_variable_over_iteration(Output_data, change_columns, grr_obj, t)
        
        # 上一次 全行参数 每行参数 对应的 loss 值
        rmse_min_last_iter = Output_data.out_rmse_list
        
        # 迭代期间 判断截止条件
        if np.min(Output_data.out_rmse_list) < odss_obj.iterative_cutoff_rmse_critical_value:
            log.logger.debug(f"==min== Output_data.out_rmse_list: {Output_data.out_rmse_list}")
            # print(f"==min== Output_data.out_rmse_list: {Output_data.out_rmse_list}")
            break
        
        """
            每次迭代输入时 清空下述变量 - 每次创建新对象耗时，所以就直接在原对象上修改属性
        """
        # 上次计算的 out_param_df 结果要保留，其余的可以删除
        Output_data.reset_some_attributes() 
        # 清空 gradient_step - 但实际上这一步没用到
        odss_obj.reset_some_attributes()      

    return grr_obj, Output_data
       

"""
    迭代过程中 持续 保存 每个参数 在 每一步的结果 - 该函数 并不影响 全方向搜索的结果，但必须运行，因为包含最终要选取的参数
"""
def record_gradient_variable_over_iteration(Output_data, change_columns, grr_obj, t):

    # 还没计算迭代前的保存
    if t == 0:
        grr_obj.param_df_record = [pd.DataFrame() for _ in range(Output_data.out_param_df.shape[0])] 
        
        grr_obj.rmse_record = [pd.DataFrame() for _ in range(Output_data.out_param_df.shape[0])]   # 二维列表
    
    change_param_cols_cnt = len(change_columns)
    
    mse_this_step = np.array(Output_data.out_rmse_list).reshape(Output_data.out_param_df.shape[0], -1)    # rmse 与 总输入参数的行数对应
    # print(f'mse_this_step : {mse_this_step}, \t, {mse_this_step.shape}')
    # 这么做是为了匹配后续生成 DataFrame的列
    mse_this_step = np.tile(mse_this_step, (1, change_param_cols_cnt))      # 在行方向上重复1次（即不重复），在列方向上重复 change_param_cols_cnt 次
    # print(f'mse_this_step : {mse_this_step}, \t, {mse_this_step.shape}')
    mse_this_step = pd.DataFrame(mse_this_step, columns=change_columns)

    assert len(mse_this_step) == Output_data.out_param_df.shape[0] , "param input shape != rmse output shape"

    for i, param_series_this_step in Output_data.out_param_df.iterrows():

        grr_obj.param_df_record[i] = grr_obj.param_df_record[i]._append(param_series_this_step, ignore_index=True)
        
        # 初始参数 也要 
        grr_obj.rmse_record[i] = grr_obj.rmse_record[i]._append(mse_this_step.iloc[i,:], ignore_index=True)

        if t not in set(grr_obj.iter_cnt):
            grr_obj.iter_cnt.append(t)
                
    return grr_obj   
    
"""
    最终所有参数 全部步骤都 迭代完后，对 grr_obj 进行处理，输出 param_at_min_rmse 和 all_params_out_df （即初始参数 + 最终选择的参数）
"""
def process_gradient_output(grr_obj, change_columns):
    # 05-21 update
    min_value = float('inf')  # 使用正无穷大作为初始值  
    min_index = None  
    
    for param_i in range(len(grr_obj.param_df_record)):     # 每个初始参数 迭代 保存
        rmse = grr_obj.rmse_record[param_i]
        
        # 获取 rmse 最小的 参数
        min_rmse_index = rmse[change_columns[0]].idxmin()     # 多列参数的 rmse 值是相同的，选一列就行
        # param_at_min_rmse = grr_obj.param_df_record[param_i].iloc[min_rmse_index,:]

        # 最后一行补0
        imagine_rmse_last_param = pd.Series(np.zeros(rmse.shape[1]), index=change_columns)
        # print('imagine_rmse_last_param :\t', imagine_rmse_last_param)
        rmse = rmse._append(imagine_rmse_last_param, ignore_index=True)   
        #rmse = rmse.append(imagine_rmse_last_param, ignore_index=True)       # - musa 服务器上的 Python 版本 不能用 _append
        # print('rmse :\t', rmse)
        
        # grr_obj.param_df_record[param_i]['rmse_g'] = rmse[change_columns[0]]       # 多列参数的 rmse 值是相同的，选一列就行
        # # 05-21 update
        # if isinstance(grr_obj.param_df_record[param_i]['rmse_g'], pd.Series):  # 检查 rmse 是否为 Series  
        #     current_min = grr_obj.param_df_record[param_i]['rmse_g'].min()  
        grr_obj.param_df_record[param_i]['ods_loss'] = rmse[change_columns[0]]       # 多列参数的 rmse 值是相同的，选一列就行
        # 05-21 update
        if isinstance(grr_obj.param_df_record[param_i]['ods_loss'], pd.Series):  # 检查 rmse 是否为 Series  
            current_min = grr_obj.param_df_record[param_i]['ods_loss'].min()  
        else:  # 如果 rmse 不是 Series，则假设它是一个数值  
            current_min = rmse 
            raise Exception('rmse Wrong Type') 
       
        if current_min < min_value:  
            min_value = current_min  
            min_index = param_i  
        
        # grr_obj.param_df_record[param_i] = process_lumped_parameters(grr_obj.param_df_record[param_i])      # 05-15 增加3个集总参数
        
        param_at_min_rmse = grr_obj.param_df_record[param_i].iloc[min_rmse_index,:]
        # print('== param_at_min_rmse == 111\n', param_at_min_rmse)
        
        if isinstance(param_at_min_rmse, (pd.Series, pd.core.series.Series)):
            param_at_min_rmse = param_at_min_rmse.to_frame().T  # T是转置操作，将列转换为行  
            
        grr_obj.param_at_min_rmse.append(param_at_min_rmse)     # 列表的每个元素是 DataFrame

    # # 05-21 update
    grr_obj.all_params_out_df = grr_obj.all_params_out_df._append(grr_obj.param_df_record[min_index].iloc[0], ignore_index=True)    # 初始第一行参数 - musa 服务器上的 Python 版本 不能用 _append
    grr_obj.all_params_out_df = grr_obj.all_params_out_df._append(grr_obj.param_at_min_rmse[min_index].iloc[0], ignore_index=True)    # 目标参数 - musa 服务器上的 Python 版本 不能用 _append
    #grr_obj.all_params_out_df = grr_obj.all_params_out_df.append(grr_obj.param_df_record[min_index].iloc[0], ignore_index=True)    # 初始第一行参数 - musa 服务器上的 Python 版本 不能用 _append
    #grr_obj.all_params_out_df = grr_obj.all_params_out_df.append(grr_obj.param_at_min_rmse[min_index].iloc[0], ignore_index=True)    # 目标参数 - musa 服务器上的 Python 版本 不能用 _append
    
    return grr_obj


"""
    获取 ods 筛选后的参数的 soh，并与 soh 分布计算的 soh 进行对比
    loss_col_name 这一列是获取损失的，可以是 'loss' 或 'ods_loss'
"""
def get_soh_based_on_ods_results(grr_obj, soh_col_name='SOH', loss_col_name='loss'):
    
    # ods 搜索后的 参数 对应的 loss - 用这些参数 去计算 SOH
    parameters_after_ods = grr_obj.param_at_min_rmse      # list, 每个元素是一行 pd.DataFrame
    
    # 只有一个元素 就 .item() 取 其数值
    loss_after_ods = [param_i[loss_col_name].item() for param_i in parameters_after_ods]
    
    soh_after_ods = [param_i[soh_col_name].item() for param_i in parameters_after_ods]
    
    # log.logger.debug(f'loss_before_ods : {loss_before_ods}')
    log.logger.debug(f'loss_after_ods : {loss_after_ods}')
    log.logger.debug(f'soh_after_ods : {soh_after_ods}')

    # print(f'loss_before_ods : {loss_before_ods}')
    # print(f'loss_after_ods : {loss_after_ods}')
    # print(f'soh_after_ods : {soh_after_ods}')
    
    soh_after_ods = np.asarray(soh_after_ods)
    
    # 注意返回一个值，不需要整个列表
    return soh_after_ods[soh_after_ods > 0].mean()      # 注意：可能有 < 0 的异常值


"""
    获取 ods 筛选后的参数的 soh，并输出参数 - 替换上面的函数
    loss_col_name 这一列是获取损失的，可以是 'loss' 或 'ods_loss'
"""
def get_soh_based_on_ods_results(grr_obj, soh_col_name='SOH', loss_col_name='loss'):
    
    # ods 搜索后的 参数 对应的 loss - 用这些参数 去计算 SOH
    parameters_after_ods = grr_obj.param_at_min_rmse      # list, 每个元素是一行 pd.DataFrame
    
    parameters_to_cal_soh = pd.DataFrame()
    for param_i in parameters_after_ods:
        parameters_to_cal_soh = concat_param_of_different_identify(parameters_to_cal_soh, param_i)
    
    
    # 只有一个元素 就 .item() 取 其数值
    loss_after_ods = parameters_to_cal_soh[loss_col_name]
    soh_after_ods = parameters_to_cal_soh[soh_col_name]
    log.logger.debug(f'loss_after_ods : {loss_after_ods}')
    log.logger.debug(f'soh_after_ods : {soh_after_ods}')
    # print(f'loss_after_ods : {loss_after_ods}')
    # print(f'soh_after_ods : {soh_after_ods}')
    
    # 筛选掉 < 0 的值
    parameters_to_cal_soh = parameters_to_cal_soh[parameters_to_cal_soh[soh_col_name] > 0]
    
    # 注意参数的 DataFrame
    return parameters_to_cal_soh


# """
#     临时画电压曲线用
# """
# def plot_volt_on_ods_results(grr_obj, dufv_obj, sipp_obj, loss_col_name='loss'):
    
#     import os
#     from call_electrochem_identify import concat_param_of_different_identify
#     from tool.draw_volt import plot_volt_err
    
#     # ods 搜索后的 参数 对应的 loss - 用这些参数 去计算 SOH
#     parameters_after_ods = grr_obj.param_at_min_rmse      # list, 每个元素是一行 pd.DataFrame
    
#     # # 只有一个元素 就 .item() 取 其数值
#     # loss_after_ods = [param_i[loss_col_name].item() for param_i in parameters_after_ods]
    
#     # print(f'loss_before_ods : {loss_before_ods}')
#     # print(f'loss_after_ods : {loss_after_ods}')
    
#     parameters_to_cal_soh = pd.DataFrame()
#     for param_i in parameters_after_ods:
#         param_i = reset_param_columns_to_fit_vol(param_i)    # 按固定的列顺序传入
        
#         # parameters_to_cal_soh = parameters_to_cal_soh._append(param_i)
        
#         parameters_to_cal_soh = concat_param_of_different_identify(parameters_to_cal_soh, param_i)
    
    
#     parameters_to_cal_soh = reset_param_columns_to_fit_vol(parameters_to_cal_soh).values
#     print(f'parameters_to_cal_soh : {parameters_to_cal_soh})')
#     # 计算 参数对应的 SOH
#     epwp_obj = create_electro_obj(dufv_obj, sipp_obj, Elecrochemical_platform_with_Parameters, parameters=parameters_to_cal_soh, soh_col_name='SOH')

#     print(f'len(parameters_to_cal_soh : {len(parameters_to_cal_soh)})')
#     # 测试 - 临时
#     sim_vol_array, loss_after_ods = epwp_obj.test(para_num=28, task_num=len(parameters_to_cal_soh), all_gather=1)
    
#     sim_vol_list = []
#     for sim_vol in sim_vol_array:
#         sim_vol_list.append(sim_vol)
    
#     # print(f'sim_vol_list : {sim_vol_list}')
#     # print(type(sim_vol_list))
#     # print(type(sim_vol_list[0]))
    
#     loss_after_ods = loss_after_ods.tolist()
#     loss_after_ods.append(0)
#     loss_labels = [str(i) for i in loss_after_ods]
#     print(f'loss_labels : {loss_labels}')
    
#     fig_save_time = time.ctime().split(':',1)
#     save_name_appendix = fig_save_time[0][-2:] + fig_save_time[1][:2]
    
#     battery_cource = 'changan'
#     save_url = fr'/home/mks/demo/mayan/ods_July/pyCxx/SOH_electro_0912/_out/_fig/{battery_cource}'
#     # if not os.path.exists(save_url):
#     #     os.makedirs(save_url)
#     save_url = os.path.join(save_url, f'fig_{save_name_appendix}.png')
#     plot_volt_err([i for i in range(len(sim_vol_list))], sim_vol_list, loss_labels, save_url)
#     print('fig plot done')