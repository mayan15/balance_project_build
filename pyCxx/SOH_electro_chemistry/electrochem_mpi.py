from mpi4py import MPI
import numpy as np
import pandas as pd
import ctypes  
import time


"""
算法说明：
除get_ods_result外其他算法任务数task_num最好可以被GPU数整除

try except待添加
"""

"""
    log.logger.debug("用于打印日志")
"""
from tool_utils.pvlog import Logger
level = "debug"
log = Logger('./logs/SOH_electro.log', level=level)

INVALID_VALUE = 65535


# 电化学模型的MPI并行计算python接口类
class MpiEC:
    # 如果已经初始化了MPI则mpi_init传入1，不需要再次初始化，且需要传入进程号rank和进程总数size值
    def __init__(self, all_gather=1):  
        # 结果收集flag，为1代表根进程手机全部进程结果
        self.all_gather = all_gather

        # 初始化 MPI
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.comm = comm
        log.logger.debug(f'(python) rank:{self.rank}  size:{self.size}  MPI init success!')
        # print(f'(python) rank:{self.rank}  size:{self.size}  MPI init success!')
 
    # 获取mpi的相关参数信息
    def get_mpi_status(self):
        return self.comm, self.rank
    
    
    # 对于多GPU任务计算每个GPU计算几个任务
    def get_batch_size(self, task_num):
        if (task_num % self.size) == 0 :
            batch_size_max = int(task_num / self.size)
        else:
            batch_size_max = int(task_num / self.size + 1)

        task_start = batch_size_max * self.rank

        # 有效结果索引
        # self.effective_result_index = np.zeros(task_num)
        if task_num - task_start > batch_size_max:
            batch_size = batch_size_max
        elif task_num > task_start:
            batch_size = task_num - task_start
        else:
            batch_size = 0

        return batch_size, batch_size_max, task_start

    # 转为2进制，为1的位代表该位对应的参数需要辨识
    def count_ones_in_binary(self, number):
        # 将整数转换为二进制字符串，去掉'0b'前缀
        binary_representation = bin(number)[2:]
        # 计算二进制字符串中'1'的个数
        count_of_ones = binary_representation.count('1')
        return count_of_ones
    
    # 数据初始化
    def data_init(self, so_path, time_arr, current_arr, voltage_arr, temperature_arr, para_num):
        # 底层算法动态链接库
        clib = ctypes.CDLL(so_path)
        self.so_lib = clib  

        # 数据及长度获取
        self.time_arr = np.array(time_arr).astype(np.float32) 
        self.current_arr = np.array(current_arr).astype(np.float32) 
        self.voltage_arr = np.array(voltage_arr).astype(np.float32)  
        self.temperature_arr = np.array(temperature_arr).astype(np.float32) 
        self.length = len(time_arr)
        self.para_num = para_num

        # 损失函数类型字典，10及以上带scd后缀的最后获取结果为加单连通域的损失值
        self.loss_func_dict = {'mse': 0, 'acc': 1, 'dqdv': 2, 'diff_max': 3, 'Rsquare': 4, 'mae': 5,
                               'mse_scd': 10, 'acc_scd': 11, 'dqdv_scd': 12, 'diff_max_scd': 13, 'Rsquare_scd': 14, 'mae_scd': 15}
        
        # 电芯类型字典
        self.cell_type_dict = {'lfp': 0, 'ncm': 1,
                               'LFP': 0, 'NCM': 1,
                               'lfp_prada': 0, 'NMC': 1,}
        
    # 转化为二维数据
    def idf_result_convert(self, arr_1d, num_row):
        # 转为二维array
        connected_component_ratio = arr_1d[-num_row:] / 3.6e6
        eta_r_ir = arr_1d[-2 * num_row:-1 * num_row]
        ohmic_ir = arr_1d[-3 * num_row:-2 * num_row]
        mse_loss = arr_1d[-4 * num_row:-3 * num_row]
        fe = arr_1d[-5 * num_row:-4 * num_row] / 3.6e6
        recharge_energy = arr_1d[-6 * num_row:-5*num_row] / 3.6e6
        ce = arr_1d[-7 * num_row:-6 * num_row] / 3.6e6
        tc = arr_1d[-8 * num_row:-7 * num_row]
        cc = arr_1d[-9 * num_row:-8 * num_row]
        loss = arr_1d[-10 * num_row:-9 * num_row]
        
        arr_2d = arr_1d[:-10 * num_row].reshape(num_row, -1)
        arr_2d = np.insert(arr_2d, 0, connected_component_ratio, axis=1)
        arr_2d = np.insert(arr_2d, 0, eta_r_ir, axis=1)
        arr_2d = np.insert(arr_2d, 0, ohmic_ir, axis=1)
        arr_2d = np.insert(arr_2d, 0, mse_loss, axis=1)
        arr_2d = np.insert(arr_2d, 0, fe, axis=1)
        arr_2d = np.insert(arr_2d, 0, recharge_energy, axis=1)
        arr_2d = np.insert(arr_2d, 0, ce, axis=1)
        arr_2d = np.insert(arr_2d, 0, tc, axis=1)
        arr_2d = np.insert(arr_2d, 0, cc, axis=1)
        arr_2d = np.insert(arr_2d, 0, loss, axis=1)

        return arr_2d
    
    # 其他宏观结果提取
    def macroscopic_result_extraction(self, all_output, num_row, num_col):
        # 数据提取，剔除各GPU无效结果后合并
        if self.all_gather == 1:            
            for ii in range(self.size):
                result_per_gpu = all_output[ii]
                arr_1d = np.array(result_per_gpu).flatten()

                # 转为二维数组
                arr_2d = np.array(arr_1d).reshape(num_row, num_col) 

                # 使用布尔索引来剔除含有INVALID_VALUE的行     
                rows_with_invalid_value = np.any(arr_2d == INVALID_VALUE, axis=1)  
                arr_2d_cleaned = arr_2d[~rows_with_invalid_value]

                # 不同GPU转化后数据汇总
                if 'result' in locals(): 
                    if len(arr_2d_cleaned) > 0:
                        result = np.concatenate((result, arr_2d_cleaned), axis=0) 
                        # print(f"(python) result in locals:{result} ")
                else:
                    result = arr_2d_cleaned
                    # print(f"(python) result not in locals:{result} ")
        # 剔除各GPU无效结果后合并，各进程返回各自GPU对应的结果，可能存在空的
        else:
            result_per_gpu = all_output
            arr_1d = np.array(result_per_gpu).flatten()

            # 转为二维数组
            arr_2d = np.array(arr_1d).reshape(num_row, num_col) 

            # 使用布尔索引来剔除含有NaN的行    
            rows_with_invalid_value = np.any(np.isnan(arr_2d), axis=1)  
            arr_2d_cleaned = arr_2d[~rows_with_invalid_value]
            result = arr_2d_cleaned

        '''
        返回结果
        对于self.gather_flag为1的每个进程都返回所有进程汇总后的结果,每个进程相同
        对于self.gather_flag为0的则每个进程返回各异的结果, 每个进程不同
        '''
        return result


    '''
    para0: 初始参数, 28位array
    dpara_up: 参数变化上界范围，为负数
    dpara_down: 参数变化下界范围，为正数
    comb_multiple: 全方向扩展倍数
    target_num: 获取参数套数
    flags_target: 变化参数标志位, 转为2进制为1的位标识需要辨识, 默认16777215代表24位都需要辨识 
    capacity_real: 容量, Ah
    cut_voltage_up: 上限电压, V
    cut_voltage_low: 下线电压, V
    loss_func: 损失值类型字符串，如'mse','mae'  
    battery_type: 电池类型字符串，如'lfp','ncm'  
    '''
    def get_ods_result(self, para0, dpara_up, dpara_down, comb_multiple=1, target_num=1, flags_target=16777215, capacity_real=120, 
                       cut_voltage_up=3.65, cut_voltage_low=2.5, loss_func='mae', battery_type='lfp'):
        # 初始化
        num_target_para = self.count_ones_in_binary(flags_target)  # 辨识的参数个数，默认24
        comb_num = pow(2, num_target_para)  # 方向数，与辨识的参数个数相关，默认16777215

        # 转为float32
        para0 = np.array(para0).astype(np.float32)
        dpara_up = np.array(dpara_up).astype(np.float32)
        dpara_down = np.array(dpara_down).astype(np.float32)

        # 结果变量buffer
        buffer_size = (self.para_num + 10) * target_num  # 参数+SOH等，默认38个*目标结果数
        ctypes_output_buffer = (ctypes.c_float * buffer_size)()  # 结果变量buffer

        # 损失类型，电池类型初始化
        loss_func_index = self.loss_func_dict[loss_func]
        battery_type_index = self.cell_type_dict[battery_type]

        # 全方向搜索法辨识参数
        start_time = time.time()
        self.so_lib.get_ods_parameters_multi_device(
            ctypes_output_buffer,
            self.time_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.current_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.voltage_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.temperature_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            para0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dpara_up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dpara_down.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(capacity_real),
            ctypes.c_float(cut_voltage_up),
            ctypes.c_float(cut_voltage_low),
            ctypes.c_int(flags_target),
            ctypes.c_int(comb_multiple),
            ctypes.c_int(comb_num),
            ctypes.c_int(self.para_num),
            ctypes.c_int(self.length), 
            ctypes.c_int(loss_func_index),
            ctypes.c_int(battery_type_index),
            ctypes.c_int(target_num),
            ctypes.c_int(self.rank),
            ctypes.c_int(self.size),
        )
        end_time = time.time()
        get_loss_time = end_time - start_time

        # 将 ctypes 数组转换为 numpy 数组
        output_buffer = np.ctypeslib.as_array(ctypes_output_buffer).astype(np.float32)

        # 把all_output分发到所有进程，这样所有进程得到的结果都一样  MPI.Allgather
        if self.all_gather == 1:
            all_output = self.comm.allgather(output_buffer)  # 含同步机制
            self.all_time = self.comm.allgather(get_loss_time)
        else:
            all_output = output_buffer
            self.all_time = get_loss_time

        # 数据提取，要分GPU
        num_row = target_num
        if self.all_gather == 1:            
            for ii in range(self.size):
                result_per_gpu = all_output[ii]
                arr_1d = np.array(result_per_gpu).flatten()
                # 转为二维数组
                arr_2d = self.idf_result_convert(arr_1d, num_row)

                # 不同GPU转化后数据汇总
                if 'result' in locals(): 
                    result = np.concatenate((result, arr_2d), axis=0) 
                    # print(f"(python) result in locals:{result} ")
                else:
                    result = arr_2d
                    # print(f"(python) result not in locals:{result} ")
        else:
            result_per_gpu = all_output
            arr_1d = np.array(result_per_gpu).flatten()
            # 转为二维数组
            result = self.idf_result_convert(arr_1d, num_row)

        # 使用sorted函数进行排序，key参数设置为lambda函数，指定根据第一列排序  
        sorted_result = sorted(result, key=lambda x: x[0])  
        
        # 选取前target_num行  
        selected_rows = np.array(sorted_result[:target_num])

        '''
        返回结果
        对于self.gather_flag为1的全部进程返回所有进程总结后的结果,每个进程相同
        对于self.gather_flag为0的则每个进程返回对应方向块结果, 每个进程不同
        '''
        return selected_rows


    '''
    p0: 初始参数, 28位array
    upbound: 参数变化上界
    lowbound: 参数变化下界
    task_num: 任务数
    particle_num: 粒子群数默认64
    para_num: 参数个数
    max_iter: 最大迭代数
    capacity_real: 电芯容量
    cut_voltage_up: 上限电压
    cut_voltage_low: 下限电压
    loss_func: 损失函数种类
    battery_type: 电池类型字符串，如'lfp','ncm'
    '''
    def get_pso_result(self, p0, upbound, lowbound, task_num=1, particle_num =64, para_num=28, max_iter=500, capacity_real=120, 
                       cut_voltage_up=3.65, cut_voltage_low=2.5, loss_func='mae', battery_type='LFP'):
        
        # # 转为float32
        # p0 = np.array(p0).astype(np.float32)
        # dpara_up = np.array(dpara_up).astype(np.float32)
        # dpara_down = np.array(dpara_down).astype(np.float32)
        
        # 一个GPU算几个任务
        batch_size, batch_size_max, task_start = self.get_batch_size(task_num)
        effect_buffer_size = batch_size * (self.para_num + 10)  # 有效结果长度
        
        # 结果变量buffer
        buffer_size = batch_size_max * (self.para_num + 10)
        ctypes_output_buffer = (ctypes.c_float * buffer_size)()  

        # 损失类型，电池类型初始化
        loss_func_index = self.loss_func_dict[loss_func]
        battery_type_index = self.cell_type_dict[battery_type]
        
        # 粒子群算法辨识参数
        start_time = time.time()
        self.so_lib.get_pso_parameters_multi_device(
            ctypes_output_buffer,
            self.time_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.current_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.voltage_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.temperature_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            upbound.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            lowbound.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(capacity_real),
            ctypes.c_float(cut_voltage_up),
            ctypes.c_float(cut_voltage_low),
            ctypes.c_int(batch_size),
            ctypes.c_int(particle_num),
            ctypes.c_int(para_num),
            ctypes.c_int(max_iter),
            ctypes.c_int(self.length),
            ctypes.c_int(loss_func_index),
            ctypes.c_int(battery_type_index),
            ctypes.c_int(self.rank),
            ctypes.c_int(self.size),
        )
        end_time = time.time()
        get_loss_time = end_time - start_time

        # 将 ctypes 数组转换为 numpy 数组
        output_buffer = np.ctypeslib.as_array(ctypes_output_buffer).astype(np.float32)

        # 将超出batch_size的转化为Nan，为了使得最终结果维度一致
        # 检查output_buffer的长度是否大于effect_buffer_size  
        if len(output_buffer) > effect_buffer_size:  
            # 将超出effect_buffer_size索引的元素设置为NaN  
            output_buffer[effect_buffer_size:] = INVALID_VALUE

        # 把all_output分发到所有进程，这样所有进程得到的结果都一样  MPI.Allgather
        if self.all_gather == 1:
            all_output = self.comm.allgather(output_buffer)  # 含同步机制
            self.all_time = self.comm.allgather(get_loss_time)
        else:
            all_output = output_buffer
            self.all_time = get_loss_time

        # 数据提取，要分GPU
        num_row = batch_size_max
        if self.all_gather == 1:            
            for ii in range(self.size):
                result_per_gpu = all_output[ii]
                arr_1d = np.array(result_per_gpu).flatten()
                
                # 转为二维数组
                arr_2d = self.idf_result_convert(arr_1d, num_row)

                # 使用布尔索引来剔除含有INVALID_VALUE的行     
                rows_with_invalid_value = np.any(arr_2d == INVALID_VALUE, axis=1)  
                arr_2d_cleaned = arr_2d[~rows_with_invalid_value]

                # 不同GPU转化后数据汇总
                if 'result' in locals(): 
                    if len(arr_2d_cleaned) > 0:
                        result = np.concatenate((result, arr_2d_cleaned), axis=0) 
                else:
                    result = arr_2d_cleaned
                    # print(f"(python) result not in locals:{result} ")
        else:
            result_per_gpu = all_output
            arr_1d = np.array(result_per_gpu).flatten()
            # 转为二维数组
            arr_2d = self.idf_result_convert(arr_1d, num_row)
            # 使用布尔索引来剔除含有INVALID_VALUE的行     
            rows_with_invalid_value = np.any(arr_2d == INVALID_VALUE, axis=1)  
            arr_2d_cleaned = arr_2d[~rows_with_invalid_value]

            result = arr_2d_cleaned

        '''
        返回结果
        对于self.gather_flag为1的每个进程都返回所有进程汇总后的结果,每个进程相同
        对于self.gather_flag为0的则每个进程返回各异的结果, 每个进程不同
        '''
        return result
    

    '''
        paras: 初始参数, 28位*task_num个元素的array
        task_num: 任务数
        para_num: 参数个数
        loss_func: 损失函数种类
        battery_type: 电池类型字符串，如'lfp','ncm'
    '''
    # 输入多套参数，返回多个loss
    def get_loss(self, paras, task_num=1, para_num=28, loss_func='mae', battery_type='LFP'):
        # 初始化
        block_size = 64 

        # 转为float32
        paras = np.array(paras).astype(np.float32)

        # 一个GPU算batch_size个任务
        batch_size, batch_size_max, task_start = self.get_batch_size(task_num)
        effect_buffer_size = batch_size  # 有效结果长度
        
        # 结果变量buffer
        buffer_size = batch_size_max
        ctypes_output_buffer = (ctypes.c_float * buffer_size)()  

        # 损失类型，电池类型初始化
        loss_func_index = self.loss_func_dict[loss_func]
        battery_type_index = self.cell_type_dict[battery_type]

        # 计算参数集loss
        start_time = time.time()
        if batch_size > 0:
            self.so_lib.get_loss_multi_device(
                ctypes_output_buffer,
                self.time_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.current_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.voltage_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.temperature_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                paras[task_start * self.para_num:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(batch_size),
                ctypes.c_int(para_num),
                ctypes.c_int(self.length),
                ctypes.c_int(loss_func_index),
                ctypes.c_int(battery_type_index),
                ctypes.c_int(block_size),
                ctypes.c_int(self.rank),
                ctypes.c_int(self.size),
            )
        end_time = time.time()
        get_loss_time = end_time - start_time

        # 将 ctypes 数组转换为 numpy 数组
        output_buffer = np.ctypeslib.as_array(ctypes_output_buffer).astype(np.float32)

        # 将超出batch_size的转化为Nan，为了使得最终结果维度一致
        # 检查output_buffer的长度是否大于effect_buffer_size  
        if len(output_buffer) > effect_buffer_size:  
            # 将超出effect_buffer_size索引的元素设置为NaN  
            output_buffer[effect_buffer_size:] = INVALID_VALUE

        # 把all_output分发到所有进程，这样所有进程得到的结果都一样  MPI.Allgather
        if self.all_gather == 1:
            all_output = self.comm.allgather(output_buffer)  # 含同步机制
            self.all_time = self.comm.allgather(get_loss_time)
        else:
            all_output = output_buffer
            self.all_time = get_loss_time

        # 依次提取各GPU数据，因为allgather要求所有结果维度一致，但是不同GPU batch_size可能不同，所以需要剔除部分数据
        num_row = batch_size_max
        num_col = 1
        result = self.macroscopic_result_extraction(all_output, num_row, num_col)
     
        return result
    
    
    '''
        paras: 初始参数, 28位*task_num个元素的array
        task_num: 任务数
        para_num: 参数个数
        loss_func: 损失函数种类
        battery_type: 电池类型字符串，如'lfp','ncm'
    '''
    # 输入多套参数，返回task_num组结果，每组结果包含 loss+有效数据长度长度+K个电压点
    def get_voltage(self, paras, task_num=1, para_num=28, loss_func='mae', battery_type='lfp'):
        # 初始化
        block_size = 64 

        # 转为float32
        paras = np.array(paras).astype(np.float32)

        # 一个GPU算batch_size个任务
        batch_size, batch_size_max, task_start = self.get_batch_size(task_num)
        effect_buffer_size = batch_size * (self.length + 2)  # 有效结果长度

        # 结果变量buffer
        buffer_size = batch_size_max * (self.length + 2)  # batch_size * (数据长度+2)， 首位为loss，第二位为长度
        ctypes_output_buffer = (ctypes.c_float * buffer_size)()  # 结果变量buffer

        # 损失类型，电池类型初始化
        loss_func_index = self.loss_func_dict[loss_func]
        battery_type_index = self.cell_type_dict[battery_type]

        # 计算参数集sim voltage
        start_time = time.time()
        if batch_size > 0:
            self.so_lib.get_sim_voltage_multi_device(
                ctypes_output_buffer,
                self.time_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.current_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.voltage_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.temperature_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                paras[task_start * self.para_num:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(batch_size),
                ctypes.c_int(para_num),
                ctypes.c_int(self.length),
                ctypes.c_int(loss_func_index),
                ctypes.c_int(battery_type_index),
                ctypes.c_int(block_size),
                ctypes.c_int(self.rank),
                ctypes.c_int(self.size),
            )
        end_time = time.time()
        get_loss_time = end_time - start_time

        # 将 ctypes 数组转换为 numpy 数组
        output_buffer = np.ctypeslib.as_array(ctypes_output_buffer).astype(np.float32)

        # 将超出batch_size的转化为Nan，为了使得最终结果维度一致
        # 检查output_buffer的长度是否大于effect_buffer_size  
        if len(output_buffer) > effect_buffer_size:  
            # 将超出effect_buffer_size索引的元素设置为NaN  
            output_buffer[effect_buffer_size:] = INVALID_VALUE  #np.nan

        # 把all_output分发到所有进程，这样所有进程得到的结果都一样  MPI.Allgather
        if self.all_gather == 1:
            all_output = self.comm.allgather(output_buffer)  # 含同步机制
            self.all_time = self.comm.allgather(get_loss_time)
        else:
            all_output = output_buffer
            self.all_time = get_loss_time

        # 依次提取各GPU数据，因为allgather要求所有结果维度一致，但是不同GPU batch_size可能不同，所以需要剔除部分数据
        num_row = batch_size_max
        num_col = self.length + 2 
        result = self.macroscopic_result_extraction(all_output, num_row, num_col)

        return result

        
    '''
        paras: 初始参数, 28位*task_num个元素的array
        task_num: 任务数
        para_num: 参数个数
        capacity_real: 电芯容量
        cut_voltage_up: 上限电压
        cut_voltage_low: 下限电压
        loss_func: 损失函数种类
        battery_type: 电池类型字符串，如'lfp','ncm'
    '''
    # 输入多套参数，返回多个capacity
    def get_capacity(self, paras, task_num=1, para_num=28, capacity_real=120, cut_voltage_up=3.65, cut_voltage_low=2.5, 
                     loss_func='mae', battery_type='LFP'):
        # 初始化
        block_size = 64 

        # 转为float32
        paras = np.array(paras).astype(np.float32)

        # 一个GPU算几个任务
        batch_size, batch_size_max, task_start = self.get_batch_size(task_num)
        effect_buffer_size = batch_size  # 有效结果长度

        # 结果变量buffer
        buffer_size = batch_size_max * 1  # 每个任务只返回一个float
        ctypes_output_buffer = (ctypes.c_float * buffer_size)()  # 结果变量buffer

        # 损失类型，电池类型初始化
        loss_func_index = self.loss_func_dict[loss_func]
        battery_type_index = self.cell_type_dict[battery_type]

        # 计算参数集loss
        start_time = time.time()
        if batch_size > 0:
            self.so_lib.get_capacity_multi_device(
                ctypes_output_buffer,
                self.time_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.current_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.voltage_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.temperature_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                paras[task_start * self.para_num:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(capacity_real),
                ctypes.c_float(cut_voltage_up),
                ctypes.c_float(cut_voltage_low),
                ctypes.c_int(batch_size),
                ctypes.c_int(para_num),
                ctypes.c_int(self.length),
                ctypes.c_int(loss_func_index),
                ctypes.c_int(battery_type_index),
                ctypes.c_int(block_size),
                ctypes.c_int(self.rank),
                ctypes.c_int(self.size),
            )
        end_time = time.time()
        get_loss_time = end_time - start_time

        # 将 ctypes 数组转换为 numpy 数组
        output_buffer = np.ctypeslib.as_array(ctypes_output_buffer).astype(np.float32)

        # 将超出batch_size的转化为Nan，为了使得最终结果维度一致
        # 检查output_buffer的长度是否大于effect_buffer_size  
        if len(output_buffer) > effect_buffer_size:  
            # 将超出effect_buffer_size索引的元素设置为NaN  
            output_buffer[effect_buffer_size:] = INVALID_VALUE

        # 把all_output分发到所有进程，这样所有进程得到的结果都一样  MPI.Allgather
        if self.all_gather == 1:
            all_output = self.comm.allgather(output_buffer)  # 含同步机制
            self.all_time = self.comm.allgather(get_loss_time)
        else:
            all_output = output_buffer
            self.all_time = get_loss_time

        # 依次提取各GPU数据，因为allgather要求所有结果维度一致，但是不同GPU batch_size可能不同，所以需要剔除部分数据
        num_row = batch_size_max
        num_col = 1
        result = self.macroscopic_result_extraction(all_output, num_row, num_col)

        return result
