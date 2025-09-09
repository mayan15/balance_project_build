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
        self.ocv_mcn = [2.83194267, 3.02503864, 3.13358356, 3.2066397, 3.27297821, 3.34025644, 3.39562989, 3.43152849,
                        3.45413792, 3.47108559, 3.48676566, 3.50333231, 3.52170005, 3.54183512, 3.56285912, 3.58345098,
                        3.60251513, 3.61963322, 3.63504582, 3.64937761, 3.66336451, 3.67767509, 3.69283086, 3.70915698,
                        3.72670331, 3.7451749, 3.76397477, 3.78240822, 3.79996426, 3.81654262, 3.83261461, 3.84942711,
                        3.86898008, 3.89218858, 3.91590439, 3.93638712, 3.95441371, 3.97207998, 3.99017945, 4.00875554,
                        4.02744207, 4.04540041, 4.06134481, 4.07389348, 4.08226546, 4.08715366, 4.09118952, 4.09837082,
                        4.11242296, 4.13502861, 4.17331878]
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
    def cal_cell_soh_soc(self, start_ocv, end_ocv, cur, plus_cnt, cell_rate_soh, cell_type=1):
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
        cal_soh = round((one_time_soh*plus_cnt)/diff_soc, 2)
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
        #bounds = [(0.001, 0.1), (0.01, 1), (100, 50000)]  # 上下界[(第一个参数的)，(第二个参数的)，(第三个参数的)]
        # bounds = ([0.0001, 0.0001, 1000], [0.1, 1, 100000])   # least_squares
        # 设置最大迭代次数
        options = {'maxiter': 10000}  # 例如，最大迭代100次
        result = minimize(self.first_order_loss_func, initial_guess, bounds=bounds, options=options)   #
        # result = least_squares(self.first_order_loss_func, initial_guess, bounds=bounds, method='trf')
        # 计算弛豫时间τ
        params = result.x
        print("优化是否成功:", result.success)
        print("优化信息:", result.message)
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


# 计算分析全部的Plus数据
def analyse_cell_plus_data(dir_plus_data, cell_type, use_year, cell_rate_soh=0):
    try:
        all_plus_rlt_list = []    # 所有电芯的测试结果--- 按照列表-字典的形式进行存储
        # 所有电芯的检测数据
        for cell_id, data_dir in dir_plus_data.items():
            cell_no = 'B'+ str(cell_id)
            cell_plus_vol_list = data_dir['plus_vol']
            cell_plus_cur_list = data_dir['plus_cur']
            cell_plus_time = data_dir['time']
            
            # 针对某一节电芯所有的测试数据 
            for i, value in enumerate(cell_plus_vol_list):
                pre_tau_soh_dir =  {'cell_id': 0, 'cell_no': '','cal_ah': 0, 'start_vol': 0 , 'end_vol': 0 , 'start_soc': 0, 'end_soc': 0, 'tau': 0, 'r0': 0,'r1': 0,'c1': 0, 'rmse': 0, 'r0_cur_down': 0}
                
                pre_tau_soh_dir['cell_id'] = cell_id
                pre_tau_soh_dir['cell_no'] = cell_no
                

                select_volts = np.array(cell_plus_vol_list[i])
                select_current = np.array(cell_plus_cur_list[i])
                select_time = cell_plus_time
                print("-----")
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
                cal_soh = CalSOH()
                cal_ah, soc_start, soc_end = cal_soh.cal_cell_soh_soc(select_volts[1300], select_volts[-1], select_current[2], 3, cell_rate_soh, cell_type)
                pre_tau_soh_dir['start_vol'] = select_volts[1300]
                pre_tau_soh_dir['end_vol'] = select_volts[-1]
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

                pre_tau_soh_dir['tau'] = tau
                pre_tau_soh_dir['r0'] = R0
                pre_tau_soh_dir['r1'] = R1
                pre_tau_soh_dir['c1'] = C1
                pre_tau_soh_dir['rmse'] = rmse
                pre_tau_soh_dir['cal_ah'] = cal_ah
                pre_tau_soh_dir['r0_cur_down'] = r0_cur_down
                pre_tau_soh_dir['start_soc'] = soc_start
                pre_tau_soh_dir['end_soc'] = soc_end

                # 保存结果
                all_plus_rlt_list.append(pre_tau_soh_dir)

                # 绘制原始数据曲线
                plt.plot(select_time, select_volts, label='ori')
                # 绘制拟合曲线
                plt.plot(x, y, label='sim', linestyle='--')
                # 添加图例和标题
                plt.legend()
                plt.title(
                    f'{"111"} tau:{round(tau, 2)}s  R0:{round(R0, 5)}Ω  R1:{round(R1, 5)}Ω  C1:{round(C1, 0)}F  rmse:{round(rmse, 2)}mV')
                plt.xlabel('Time (s)')
                plt.ylabel('Voltage (V)')

                # 显示图形
                plt.savefig(f'./Relax_B{1}_{1}.png')
                # plt.show()
                plt.close()
        return  all_plus_rlt_list
    except Exception as e:
        log.logger.error(f"analyse_cell_plus_data error: {traceback.print_exc()}")
    


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


def run(df_raw, data_clean_rlt, picture_save_path):
    """
       # df_raw: 原始数据
       # picture_save_path: 图片保存路径
       # 返回值：
       # rlt_res: 结果字典
    """
    try:
        rlt_res = {
        "code_id": 2,
        "describe": "consistentcy",
        "out": {},
        "summary": [],
        "table": [],
        "ErrorCode": [0, 0, '']
    }
        st = time.time()
       
        # BMS传感器有效性总评结果
        # add_out_dir_info(rlt_res, 'BMS传感器有效性总评分', sensor_socre, '', '')
        # add_out_dir_info(rlt_res, 'BMS电压传感器有效性评分', vol_sensor_check_socker, vol_sensor_valid_rlt['summary'], vol_sensor_valid_rlt['advice'])
        # # add_out_dir_info(rlt_res, 'BMS温度传感器有效性评分', temp_sensor_check_socker, temp_sensor_valid_rlt['summary'], temp_sensor_valid_rlt['advice'])
        # add_out_dir_info(rlt_res, 'BMS电流传感器有效性评分', cur_sensor_check_socker, current_sensor_valid_rlt['summary'], current_sensor_valid_rlt['advice'])
        return rlt_res   
    except Exception as e:
        log.logger.error(f"first_order_fit error: {traceback.print_exc()}")
        # add_out_dir_info(rlt_res, 'BMS传感器有效性总评分', 'N/A', '', '')
        # add_out_dir_info(rlt_res, 'BMS电压传感器有效性评分', vol_sensor_check_socker, vol_sensor_valid_rlt['summary'], vol_sensor_valid_rlt['advice'])
        # # add_out_dir_info(rlt_res, 'BMS温度传感器有效性评分', temp_sensor_check_socker, temp_sensor_valid_rlt['summary'], temp_sensor_valid_rlt['advice'])
        # add_out_dir_info(rlt_res, 'BMS电流传感器有效性评分', cur_sensor_check_socker, current_sensor_valid_rlt['summary'], current_sensor_valid_rlt['advice'])
        return rlt_res
          
