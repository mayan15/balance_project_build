# @Time          : 2023/8/28 15:00
# @Author        : Yan Ma
# @File          : pyCxx
# @Software      : Spyder
# Version        : Python3.9
# Company        : .

# 导入库
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.optimize import curve_fit

from tool_utils.pvlog import Logger

level = "debug"
log = Logger('./logs/fist_order.log', level=level)


import time
import traceback

class CalSOH:
    def __init__(self):
        # self.ocv_mcn = [2.83194267, 3.02503864, 3.13358356, 3.2066397, 3.27297821, 3.34025644, 3.39562989, 3.43152849,
        #                 3.45413792, 3.47108559, 3.48676566, 3.50333231, 3.52170005, 3.54183512, 3.56285912, 3.58345098,
        #                 3.60251513, 3.61963322, 3.63504582, 3.64937761, 3.66336451, 3.67767509, 3.69283086, 3.70915698,
        #                 3.72670331, 3.7451749, 3.76397477, 3.78240822, 3.79996426, 3.81654262, 3.83261461, 3.84942711,
        #                 3.86898008, 3.89218858, 3.91590439, 3.93638712, 3.95441371, 3.97207998, 3.99017945, 4.00875554,
        #                 4.02744207, 4.04540041, 4.06134481, 4.07389348, 4.08226546, 4.08715366, 4.09118952, 4.09837082,
        #                 4.11242296, 4.13502861, 4.17331878]
        self.ocv_mcn = [3.0991, 3.2803, 3.3697, 3.4014, 3.4151, 3.4294, 3.4475, 3.4671,
                        3.4863, 3.5038, 3.5194, 3.5344, 3.549, 3.5625, 3.5741, 3.5836,
                        3.5915, 3.5986, 3.605, 3.6112, 3.6175, 3.6241, 3.631, 3.6381,
                        3.6461, 3.6552, 3.6664, 3.6812, 3.7007, 3.7227, 3.7432, 3.7624,
                        3.7816, 3.8009, 3.8208, 3.8408, 3.8613, 3.8819, 3.9026, 3.9235,
                        3.9449, 3.9668, 3.9894, 4.0127, 4.0366, 4.0617, 4.087, 4.1131,
                        4.141, 4.1711, 4.2094]
        self.soc_mcn = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
                        34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0,
                        66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0,
                        98.0, 100.0]

        self.ocv_lfp = [2.292, 2.935, 3.076, 3.162, 3.195, 3.199, 3.201, 3.202, 3.213, 3.228, 3.24, 3.251, 3.257, 3.264,
                        3.271, 3.277, 3.281, 3.283, 3.285, 3.285, 3.286, 3.286, 3.287, 3.287, 3.287, 3.288, 3.289, 3.289, 3.29, 3.291, 3.294, 3.303
                        ,3.319, 3.326, 3.327, 3.327, 3.328, 3.328, 3.328, 3.328, 3.328, 3.328, 3.328, 3.329, 3.329, 3.329, 3.33
                        ,3.33, 3.33, 3.332, 3.369]
        self.soc_lfp = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48
            ,50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]
        
        self.ocv_to_soc = np.poly1d(np.polyfit(self.ocv_mcn, self.soc_mcn, 4))
        self.soc_to_ocv = np.poly1d(np.polyfit(self.soc_mcn, self.ocv_mcn, 4))

        self.ocv_to_soc_lfp = np.poly1d(np.polyfit(self.ocv_lfp, self.soc_lfp, 7))
        self.soc_to_ocv_lfp = np.poly1d(np.polyfit(self.soc_lfp, self.ocv_lfp, 7))


    def get_lfp_soc_for_ocv(self, vol):
        iii = 0
        for index, item in enumerate(self.ocv_lfp):
            if vol < item:
                iii = index
                break
        cal_soc = 0

        if iii > 0:
            # 20250827 my修改
            soc_start = self.soc_lfp[iii - 1]
            numerator = 2 * (vol - self.ocv_lfp[iii-1])
            denominator = (self.ocv_lfp[iii] - self.ocv_lfp[iii - 1])
            cal_soc = soc_start + numerator / denominator
            # cal_soc = self.soc_lfp[iii - 1] + 2 * (self.ocv_lfp[iii] - vol) / (
            #             self.ocv_lfp[iii] - self.ocv_lfp[iii - 1])
        return cal_soc

    # cur: 2929
    def cal_cell_soh_soc(self, start_ocv, end_ocv, cur, pulse_cnt, cell_rate_soh, cell_type=1):
        one_time_soh = cur/60
        if cell_type == 2:   # mcn  三元
            soc_start = self.ocv_to_soc(start_ocv)
            soc_end = self.ocv_to_soc(end_ocv)
            diff_soc = np.abs(soc_start - soc_end)
            # self.soc = self.ocv_to_soc(end_ocv)
        else:   # lfp   铁锂
            soc_start = self.get_lfp_soc_for_ocv(start_ocv)
            soc_end = self.get_lfp_soc_for_ocv(end_ocv)
            diff_soc = np.abs(soc_start - soc_end)
            # self.soc = self.soc_lfp[self.get_lfp_soc_for_ocv(end_ocv)]
        cal_soh = round((one_time_soh*pulse_cnt)/diff_soc, 2)
        if cal_soh > cell_rate_soh:
            cal_soh = 0
        return [cal_soh, round(soc_start, 1), round(soc_end, 1)]


class FirstOrderFit:
    def __init__(self, time_s, vol_V, cur_A, dt_s, vol_start, vol_end, weight_array):
        self.time = time_s
        self.vols = vol_V
        self.cur = cur_A
        self.dt = dt_s
        self.V0 = vol_start
        self.V1 = vol_end
        self.weight = weight_array

    # 定义一阶电压响应函数
    def first_order_model(self, params):
        R0, R1, C1 = params
        t = self.time
        I = self.cur
        dt = self.dt
        V0 = self.V0
        V1 = self.V1

        # I为 0的情况
        exp_term = np.exp(-dt / (R1 * C1))    

        # 处理 I 不等于 0 的情况
        V_non_zero_I = V0 - I * R0 - I * R1 * (1 - exp_term)

        # 处理 I 等于 0 的情况
        # 注意：在数值计算中，直接比较浮点数是否等于0是不安全的，因为可能存在极小的误差
        # 这里我们使用一个小的阈值来判断 I 是否“足够接近”0
        threshold = 1e-10  # 可以根据具体情况调整这个阈值
        I_is_zero = np.abs(I) < threshold

        # share_R0 = np.zeros_like(dt)
        # share_R0[np.where(dt == 0)] = R0 / (R0 + R1)

        #V_zero_I = V1 + (V0 - V1) * (R1 / (R0 + R1) * exp_term)

        V_zero_I = V1 + (V0 - V1) * exp_term

        # 使用 np.where 来根据 I 的值选择 V 的值
        V = np.where(I_is_zero, V_zero_I, V_non_zero_I) 
        return V

    # 定义一阶电压损失函数，可更改损失值形式
    def first_order_loss_func(self, params):
        V = self.first_order_model(params)
        Vt = self.vols
        loss = np.sum(((V - Vt) * 1000)**2)
        #loss = np.sum(((V - Vt)*1000) ** 2)    # min
        #loss = np.sum(self.weight * (((V - Vt) * 1000) ** 2))
        return loss

    # 均方根
    def calculate_rmse(self, data):
        sum_squared = sum([x ** 2 for x in data])
        n = len(data)

        rmse = math.sqrt(sum_squared / n)
        mse = sum_squared / n
        return rmse, mse

    # 参数辨识
    def first_order_fit(self):
        # 使用curve_fit函数进行一阶参数拟合
        initial_guess = np.array([0.0001, 0.0001, 60000])  # 初始参数猜测值，可变更
        # bounds = [(0.0001, 0.1), (0.0001, 0.1), (100, 100000)]  # 上下界[(第一个参数的)，(第二个参数的)，(第三个参数的)]
        bounds = [(0.0001, 0.1), (0.0001, 1), (100, 100000)]  # 上下界[(第一个参数的)，(第二个参数的)，(第三个参数的)]
        # bounds = [(0.001, 0.1), (0.01, 1), (100, 50000)]  # 上下界[(第一个参数的)，(第二个参数的)，(第三个参数的)]
        # bounds = ([0.0001, 0.0001, 1000], [0.1, 1, 100000])   # least_squares
        # 设置最大迭代次数
        options = {'maxiter': 10000}  # 例如，最大迭代100次
        result = minimize(self.first_order_loss_func, initial_guess, bounds=bounds, options=options)   #
        # result = least_squares(self.first_order_loss_func, initial_guess, bounds=bounds, method='trf')
        # 计算弛豫时间τ
        params = result.x
        
        # print("优化是否成功:", result.success)
        # print("优化信息:", result.message)
        R0 = params[0]
        R1 = params[1]
        C1 = params[2]

        tau = round(R1 * C1, 2)
        R1 = round(R1, 4)
        C1 = round(C1, 1)

        # 生成拟合曲线值
        x_fit = np.array(self.time)
        y_fit = self.first_order_model(params)

        # 计算差值的绝对值
        diff_abs = np.abs(np.array(y_fit)*1000 - np.array(self.vols)*1000)
        # 找到最大差值的索引
        max_diff_index = np.argmax(diff_abs)
        # 计算最大差值
        max_diff = diff_abs[max_diff_index]
        max_diff = round(max_diff, 1)
        mean_diff = round(np.mean(diff_abs), 1)
        # 计算差值平均值
        rmse, mse = self.calculate_rmse(diff_abs)
        rmse = round(rmse, 1)
        mse = round(mse, 1)

        return tau, R0, R1, C1, mean_diff, max_diff, rmse, x_fit, y_fit

# 弛豫曲线一阶拟合
def _relax_vol_fit_ecm(time_s, vol_V, cur_A, dt_s, vol_start, vol_end, weight_array):

    # 拟合结果, tau时间常数单位为s，mean_diff为拟合平均压差单位mV，max_diff为拟合最大压差单位mV，rmse单位mV
    handle = FirstOrderFit(time_s, vol_V, cur_A, dt_s, vol_start, vol_end, weight_array)
    tau, R0, R1, C1, mean_diff, max_diff, rmse, x, y = handle.first_order_fit()
    return tau, R0, R1, C1, x, y, rmse


# 计算分析全部的pulse数据
def analyse_cell_pulse_data(dir_pulse_data, cell_type, use_year, cell_rate_soh=0):
    try:
        # all_pulse_rlt_list = []    # 所有电芯的测试结果--- 按照列表-字典的形式进行存储
        all_tau_dir = {}    # 所有电芯的测试结果--- 按照字典-字典的形式进行存储
        # 以下存放每一节电芯计算结果最合适的值
        cell_id_list = []
        best_tau_list = []
        best_tau_start_vol_list = []
        best_r0_list = []
        best_r1_list = []
        best_r0_cur_down_list = []
        # 所有电芯的检测数据
        for cell_id, data_dir in dir_pulse_data.items():
            tau_value_dir =  {'start_vol': [] , 'end_vol': [] , 'start_soc': [], 'end_soc': [], 'tau': [], 'r0': [],'r1': [],'c1': [], 'rmse': [], 'r0_cur_down': [], 'best_index': -1}
            # cell_no = 'B'+ str(cell_id)
            cell_pulse_vol_list = data_dir['pulse_vol']
            cell_pulse_cur_list = data_dir['pulse_cur']
            cell_pulse_time = data_dir['time']
            
            # 针对某一节电芯所有的测试数据 
            for i, value in enumerate(cell_pulse_vol_list):
                select_volts = np.array(cell_pulse_vol_list[i])
                select_current = np.array(cell_pulse_cur_list[i])
                select_time = cell_pulse_time
                # 对数据列进行处理
                dt = select_time[0]
                dt_list = []
                t_last = 0

                cur_last = select_current[0]
                index_start_list = []
                index_end_list = []
                index_start_list.append(0)

                # 电流突变增加权重
                weight_list = []
                max_weight = 10  # 这个地方可以改，用来调节权重

                # 数据分段（放电/搁置/放电/搁置），获取每一小段的dt以及起始电压结束电压对应的index
                for ii in range(len(select_time)):
                    s_cur = select_current[ii]
                    if s_cur == cur_last:
                        dt = select_time[ii] - t_last
                        dt_list.append(dt)
                        weight_list.append(1)
                    elif np.isnan(s_cur):
                        continue
                    else:
                        dt_list.append(0)
                        index_start_list.append(ii)
                        index_end_list.append(ii-1)
                        t_last = select_time[ii]
                        cur_last = s_cur
                        weight_list[-1] = max_weight * 10  # 每段首尾增加loss权重
                        weight_list.append(max_weight)  # 每段首尾增加loss权重

                
                # 计算内阻
                start_index_vol = [select_volts[i] for i in index_start_list]
                end_index_vol = [select_volts[i] for i in index_end_list]
                # 1. 取奇数索引的值（list1）
                vals1 = end_index_vol[1::2]   
                # 2. 取偶数索引的值（list2）
                vals2 = start_index_vol[2::2]   
                # 3. 计算
                r0_cur_down = [(a - b) / select_current[2] for a, b in zip(vals2, vals1)]
                
                index_end_list.append(len(select_time)-1)
                select_dt = np.array(dt_list)

                index_start_array = np.array(index_start_list)
                index_end_array = np.array(index_end_list)

                weight_list[0] = max_weight  # 整段首尾增加loss权重
                weight_list[-1] = max_weight  # 整段首尾增加loss权重
                weight_array = np.array(weight_list)

                if max(select_volts) > 1000:
                    select_volts = select_volts / 1000

                vol_start_list = []
                vol_end_list = []

                for ii in range(len(select_time)):
                    indices_less_than_ii = np.where(np.logical_or(index_start_array-1 < ii, index_start_array == 0))[0]
                    index_start = int(index_start_array[indices_less_than_ii[-1]])
                    index_end = int(index_end_array[indices_less_than_ii[-1]])
                # 不考虑电流对ocv影响版本，这个地方可以改，因为充放电过程SOC是变化的，对应vol_start_list中的值在有电流情况下作为OCV进行计算可能有问题
                # 已在下面优化成线性插值法得到每个点OCV，但是实际应该根据SOC查表得到真实OCV，还有优化空间
                    # 有电流情况插值得到变化的OCV
                    if abs(select_current[ii]) > 0.1:
                        # 考虑充放电对ocv影响，暂时没有做保护，最后一段不是静置应该会报错
                        ocv1 = select_volts[index_start]
                        ocv2 = select_volts[int(index_end_array[indices_less_than_ii[-1] + 1])]

                        dv = ocv2 - ocv1
                        di_ratio = (ii - index_start)/(index_end - index_start)
                        dv = 0
                        vol_start_list.append(select_volts[index_start] + dv * di_ratio)
                        vol_end_list.append(select_volts[index_end])
                    # 没电流情况根据弛豫结束电压适当增减，优化弛豫过程OCV，数值上拟合效果更好
                    else:
                        vol_start_list.append(select_volts[index_start])
                        vol_end_list.append(select_volts[index_end])  # 这个地方可以改，因为弛豫时间较短没有达到实际OCV，可以适当加电压（如果是充电得改成减电压）
            
                select_volt_start = np.array(vol_start_list)
                select_volt_end = np.array(vol_end_list)

                # 计算SOH
                # cal_soh = CalSOH()
                # cal_ah, soc_start, soc_end = cal_soh.cal_cell_soh_soc(select_volts[1300], select_volts[-1], select_current[2], 3, cell_rate_soh, cell_type)
                
                tau_value_dir['start_vol'].append(select_volts[0])
                tau_value_dir['end_vol'].append(select_volts[-1])

                if max(select_current) > 100:
                    if select_volt_end[1] < select_volt_start[1]:   # discharge
                        select_current = select_current/100
                    else:
                        select_current = -(select_current/100)      # charge current is negative
    
                # x,y为拟合曲线  #tau, R0, R1, C1, x, y, rmse
                tau, R0, R1, C1, x, y, rmse = _relax_vol_fit_ecm(select_time[0:int(len(select_time))],
                                                                    select_volts[0:int(len(select_time))],
                                                                    select_current[0:int(len(select_time))],
                                                                    select_dt[0:int(len(select_time))],
                                                                    select_volt_start,
                                                                    select_volt_end,
                                                                    weight_array)

                 
                tau_value_dir['tau'].append(tau)
                tau_value_dir['r0'].append(round(R0*1000,2))  # 单位转为mΩ
                tau_value_dir['r1'].append(round(R1*1000,2))  # 单位转为mΩ
                tau_value_dir['c1'].append(C1)
                tau_value_dir['rmse'].append(rmse)
                tau_value_dir['r0_cur_down'].append(round(r0_cur_down[0]*1000,2))

                # # 绘制原始数据曲线
                # plt.plot(select_time, select_volts, label='ori')
                # # 绘制拟合曲线
                # plt.plot(x, y, label='sim', linestyle='--')
                # # 添加图例和标题
                # plt.legend()
                # plt.title(
                #     f'{"111"} tau:{round(tau, 2)}s  R0:{round(R0, 5)}Ω  R1:{round(R1, 5)}Ω  C1:{round(C1, 0)}F  rmse:{round(rmse, 2)}mV')
                # plt.xlabel('Time (s)')
                # plt.ylabel('Voltage (V)')

                # # 显示图形
                # plt.savefig(f'./Relax_B{1}_{1}.png')
                # # plt.show()
                # plt.close()
            # 从当前电芯所有结果中，筛选出最合适的结果出来：
            if cell_type == "LFP":
                VOL_LIMIT = 3.200     
            else:
                VOL_LIMIT = 3.660 
            # 找出与 VOL_LIMIT 差值最小的 start_vol 值的索引
            start_vol_values = tau_value_dir['start_vol']
            if start_vol_values:
                differences = np.abs(np.array(start_vol_values) - VOL_LIMIT)
                min_index = np.argmin(differences)
                tau_value_dir['best_index'] = min_index
            
            all_tau_dir[cell_id] = tau_value_dir 
            
            cell_id_list.append(cell_id)
            best_tau_list.append(tau_value_dir['tau'][tau_value_dir['best_index']])
            best_tau_start_vol_list.append(tau_value_dir['start_vol'][tau_value_dir['best_index']])
            best_r0_list.append(tau_value_dir['r0'][tau_value_dir['best_index']])
            best_r1_list.append(tau_value_dir['r1'][tau_value_dir['best_index']])
            best_r0_cur_down_list.append(tau_value_dir['r0_cur_down'][tau_value_dir['best_index']])

        return  all_tau_dir, cell_id_list, best_tau_list, best_tau_start_vol_list, best_r0_list, best_r1_list, best_r0_cur_down_list

    except Exception as e:
        log.logger.error(f"analyse_cell_pulse_data error: {traceback.print_exc()}")
        raise RuntimeError(f'analyse_cell_pulse_data函数出错,{e}') from e
    


def add_out_dir_info(rlt_res, key, value, confidence, explanation):
    """
    向 rlt_res 中添加 key value and confidence explanation

    参数：
    rlt_res: 结果字典
    key: '充电起始总电压'或'充电结束总电压'
    value: 总电压值
    confidence: 可信度信息 [-10, 10]  正值代表可信度，值越大，可信度越高， 负值表示异常， 值越小，异常等级越高
    explanation : 对可信度值的解释
    """
    # 初始化指定的键
    rlt_res['out'][key] = []
    rlt_res['out'][key].append(value)
    rlt_res['out'][key].append(confidence)
    rlt_res['out'][key].append(explanation)

def invalid_result_handle(rlt_res):
    """处理无效结果"""
    add_out_dir_info(rlt_res, '内阻计算全部结果值', 'N/A', '', '')
    add_out_dir_info(rlt_res, '内阻计算电芯号', 'N/A', '', '')
    add_out_dir_info(rlt_res, '内阻计算电芯起始电压值', 'N/A', '', '')
    add_out_dir_info(rlt_res, '内阻计算直流内阻值', 'N/A', '', '')
    add_out_dir_info(rlt_res, '内阻计算tau值', 'N/A', '', '')
    add_out_dir_info(rlt_res, '内阻计算R0值', 'N/A', '', '')
    add_out_dir_info(rlt_res, '内阻计算R1值', 'N/A', '', '')
    add_out_dir_info(rlt_res, '内阻值异常电芯号', 'N/A', '', '')
    add_out_dir_info(rlt_res, '内阻计算异常说明', '详见fist_order.log', '', '')
    add_out_dir_info(rlt_res, '说明', '内阻测试没有有效数据或计算结果。', '', '')
    add_out_dir_info(rlt_res, '建议', '暂无建议。', '', '')

def run(df_raw, data_clean_rlt):
    """
       # df_raw: 原始数据
       # picture_save_path: 图片保存路径
       # 返回值：
       # rlt_res: 结果字典
    """

    try:
        rlt_res = {
        "code_id": 1,
        "describe": "pulse",
        "out": {},
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }
        
        summary= '/'
        advice= '/'

        st = time.time()
        all_tau_rlt, cell_id_list, best_tau_list, best_tau_start_vol_list, best_r0_list, best_r1_list, best_r0_cur_down_list = analyse_cell_pulse_data(df_raw, data_clean_rlt['out']['battery_type'][0], 5, data_clean_rlt['out']['battery_capacity'][0])

        # # 20250916 my添加 手动制造异常结果
        # best_r0_cur_down_list[0] = np.max(best_r0_cur_down_list) * 3

        # 判断结果值对比可信度
        if len(best_tau_start_vol_list) > 1:
            # 找出最大和最小的内阻值，以及对应的index 
            max_r0_cur_down_index = np.argmax(best_r0_cur_down_list)
            min_r0_cur_down_index = np.argmin(best_r0_cur_down_list)
            median_r0_cur_down = np.median(best_r0_cur_down_list)

            # 设定阈值，并找出超过中值内阻值的50%，为异常电芯
            max_r0_cur_down_threshold = 0.5
            dcir_abnormal_cell_no_list = [id for r,id in zip(best_r0_cur_down_list, cell_id_list) if r > median_r0_cur_down * (1 + max_r0_cur_down_threshold)]
            dcir_abnormal_cell_no_str = ','.join(str(i) for i in dcir_abnormal_cell_no_list) if len(dcir_abnormal_cell_no_list) > 0 else '/'

            dec_r0_cur_down = round(best_r0_cur_down_list[max_r0_cur_down_index] - best_r0_cur_down_list[min_r0_cur_down_index],2)
            ratio_dec_r0_cur_down = dec_r0_cur_down / best_r0_cur_down_list[max_r0_cur_down_index]
            
            vol_dec = np.max(best_tau_start_vol_list) - np.min(best_tau_start_vol_list)
            if vol_dec < 0.015:  # 差值小于0.01，认为结果可信
               if len(dcir_abnormal_cell_no_list) == 0:  
                    summary= f'进行内阻测试的{len(cell_id_list)}节电芯中，内阻差值{dec_r0_cur_down}mΩ，最大内阻值{best_r0_cur_down_list[max_r0_cur_down_index]}mΩ，最小内阻值{best_r0_cur_down_list[min_r0_cur_down_index]}mΩ, 内阻一致性正常。'  
                    advice = '暂无建议。'
            #    elif ratio_dec_r0_cur_down < 0.5:
            #         summary= f'进行内阻测试的{len(cell_id_list)}节电芯中，内阻差值{dec_r0_cur_down}mΩ，最大内阻值{best_r0_cur_down_list[max_r0_cur_down_index]}mΩ，最小内阻值{best_r0_cur_down_list[min_r0_cur_down_index]}mΩ，内阻一致性较差。'
            #         advice = '暂无建议'
               else:
                    summary= f'进行内阻测试的{len(cell_id_list)}节电芯中，内阻差值{dec_r0_cur_down}mΩ，最大内阻值{best_r0_cur_down_list[max_r0_cur_down_index]}mΩ，最小内阻值{best_r0_cur_down_list[min_r0_cur_down_index]}mΩ，内阻一致性差。'
                    advice = f'电池组内阻一致性存在问题，建议对内阻较大的异常电芯进行二次测试确认，若仍存在，建议更换。'
            else:
                summary = f'进行内阻测试的{len(cell_id_list)}节电芯中，内阻差值{dec_r0_cur_down}mΩ，最大内阻值{best_r0_cur_down_list[max_r0_cur_down_index]}mΩ，最小内阻值{best_r0_cur_down_list[min_r0_cur_down_index]}mΩ。'
                advice = f'电池组电芯间电压值存在差别，该内阻值仅供参考。'
        else:
            dcir_abnormal_cell_no_str = '/'
        # 部分结果转为字符串
        cell_id_list_str = ','.join(str(i) for i in cell_id_list)
        best_tau_start_vol_list_str = ','.join(str(i) for i in best_tau_start_vol_list)
        best_r0_cur_down_list_str = ','.join(str(i) for i in best_r0_cur_down_list)
        best_tau_list_str = ','.join(str(i) for i in best_tau_list)
        best_r0_list_str = ','.join(str(i) for i in best_r0_list)
        best_r1_list_str = ','.join(str(i) for i in best_r1_list)

        # tau值计算结果
        add_out_dir_info(rlt_res, '内阻计算全部结果值', all_tau_rlt, '', '')
        add_out_dir_info(rlt_res, '内阻计算电芯号', cell_id_list_str, '','')
        add_out_dir_info(rlt_res, '内阻计算电芯起始电压值', best_tau_start_vol_list_str, '','')
        add_out_dir_info(rlt_res, '内阻计算直流内阻值', best_r0_cur_down_list_str, '','')
        add_out_dir_info(rlt_res, '内阻计算tau值', best_tau_list_str, '','')
        add_out_dir_info(rlt_res, '内阻计算R0值', best_r0_list_str, '','')
        add_out_dir_info(rlt_res, '内阻计算R1值', best_r1_list_str, '','')
        add_out_dir_info(rlt_res, '内阻值异常电芯号', dcir_abnormal_cell_no_str, '', '')
        add_out_dir_info(rlt_res, '内阻计算异常说明', '', '', '')
        add_out_dir_info(rlt_res, '说明', summary, '','')
        add_out_dir_info(rlt_res, '建议', advice, '','')
        log.logger.debug(f"tau calculate time: {round(time.time()-st,2)} seconds")
        return rlt_res   
    except Exception as e:
        rlt_res['ErrorCode'][0] = 1001
        log.logger.error(f"first_order_fit error: {traceback.print_exc()}")
        invalid_result_handle(rlt_res)
        return rlt_res
          
