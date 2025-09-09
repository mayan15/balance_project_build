import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)      # 屏蔽 SetingWithCopyWarning

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体 
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams["figure.max_open_warning"] = 150      # 以确保Matplotlib可以同时打开最多150个文件

"""
    log.logger.debug("用于打印日志")
"""
from tool_utils.pvlog import Logger, level_SOH_electro
log = Logger('./logs/SOH_electro.log', level=level_SOH_electro)

"""
    -- SOH筛选参数
    mean_fit: SOH 分布均值
    std_fit: SOH 分布方差
    param_df_list: 不同电池的 参数 list
    bin_num: hist图形的柱子数
    learning_rate: 学习率
    epochs: 迭代次数
        
"""

# 01-23 15：32更新
class Screen_Param_By_SOH_Distribution(object):
    
    def __init__(self, mean_fit, std_fit, param_df_list, bin_num, learning_rate, epochs, if_plot, soh_hist_url):
        self.mean_fit = mean_fit
        self.std_fit = std_fit
        self.param_df_list = param_df_list
        self.bin_num = bin_num
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.if_plot = if_plot
        self.soh_hist_url = soh_hist_url
    
    def run_this_class(self):
        
        list_of_df_param_after_max_density_of_SOH = []
        # for i, df_param in enumerate(self.param_df_list):
        #     df_max_density_of_SOH = self.select_param_by_soh(df_param)
        #     list_of_df_param_after_max_density_of_SOH.append(df_max_density_of_SOH)
        
        # 01-23 15：32更新 输出 拟合的 std
        soh_fit_std_list = []
        for i, df_param in enumerate(self.param_df_list):
            df_max_density_of_SOH, soh_fit_std = self.select_param_by_soh(df_param)
            list_of_df_param_after_max_density_of_SOH.append(df_max_density_of_SOH)
            soh_fit_std_list.append(soh_fit_std)
        
        return list_of_df_param_after_max_density_of_SOH, soh_fit_std_list
                                                    
    
    def select_param_by_soh(self, df_before_soh_filtering):
        
        params = [self.mean_fit, self.std_fit]
        efficacious_rows = len(df_before_soh_filtering)
        bin_values, bin_edges = np.histogram(df_before_soh_filtering['SOH'], bins=self.bin_num)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        params_fit = gradient_descent(params, bin_centers, bin_values/efficacious_rows, self.learning_rate, self.epochs)
        
        mean = params_fit[0]
        std = params_fit[1]

        if self.if_plot:
            self.plot_soh(df_before_soh_filtering, efficacious_rows, mean, std)
        
        # # 筛选出距离均值最近的数据 - 01-24 16：28更新双重 mean_dev_range
        # mean_dev_range = 0.5
        # df_max_density_of_SOH = df_before_soh_filtering[(df_before_soh_filtering['SOH'] > mean-mean_dev_range) & (df_before_soh_filtering['SOH'] < mean+mean_dev_range)]
        
        # if len(df_max_density_of_SOH) < 1:
        #     mean_dev_range = mean_dev_range * 2
        #     df_max_density_of_SOH = df_before_soh_filtering[(df_before_soh_filtering['SOH'] > mean-mean_dev_range) & (df_before_soh_filtering['SOH'] < mean+mean_dev_range)]
        
        # if len(df_max_density_of_SOH) < 1:
        #     closest_index = abs(df_before_soh_filtering['SOH'] - mean).argmin()
        #     df_max_density_of_SOH = df_before_soh_filtering.iloc[closest_index]
            
        #     if type(df_max_density_of_SOH) in [pd.Series]:
        #         # df_max_density_of_SOH = df_max_density_of_SOH.to_frame()
        #         # # 01-25 16:34 更新, 如果不转置, 就会将列名变为 index
        #         df_max_density_of_SOH = df_max_density_of_SOH.to_frame().T
            
        '''
            # 筛选出距离均值最近的数据 -  01-25 16:34 更新, 选择距离最近的前 sort_num 个点 
            # 替换上面的结果
        '''
        sort_num = 2
        closest_index = abs(df_before_soh_filtering['SOH'] - mean).argsort()[:sort_num]
        df_max_density_of_SOH = df_before_soh_filtering.iloc[closest_index]
        
        if type(df_max_density_of_SOH) in [pd.Series]:
            # # 01-25 16:34 更新, 如果不转置, 就会将列名变为 index
            df_max_density_of_SOH = df_max_density_of_SOH.to_frame().T
            
        
        # 索引重置
        df_max_density_of_SOH = df_max_density_of_SOH.reset_index(drop=False)    # drop=False, 是为了上一步的索引会保存
        # df_max_density_of_SOH = df_max_density_of_SOH.reset_index(drop=True)     # drop=False, 是为了上一步的索引会保存
        
        return df_max_density_of_SOH, std
    
    
    def plot_soh(self, df_before_soh_filtering, efficacious_rows, mean, std):
        
        df_before_filtering = df_before_soh_filtering['SOH']

        # 正态分布图
        # x_fit = np.linspace(df_before_filtering.min(), df_before_filtering.max(), 100)
        x_fit = np.linspace(df_before_filtering.min(), df_before_filtering.max(), efficacious_rows)
        y_fit = normal_distribution(x_fit, mean, std)

        # 作图
        fig = plt.figure(dpi=400)
        ax1 = fig.add_subplot(111)
        # ax1.hist(df_before_filtering, bins = 30, color = 'blue', edgecolor = 'black', density = False, alpha = 1, label = 'Histogram')
        ax1.hist(df_before_filtering, bins = self.bin_num, color = 'blue', edgecolor = 'black', density = False, alpha = 1, label = 'Histogram')

        ax2 = ax1.twinx()
        ax2.plot(x_fit, y_fit, '-r', linewidth = 2, label=f'mean={round(mean,2)}\n std={round(std,2)}')
        # fig.legend(loc='center')
        ax2.legend(loc='upper right')
        
        ax1.set_xlabel(r'soh / %')
        ax1.set_ylabel(r'value counts')
        
        ax2.set_ylabel(r'frequency / %')
        
        fig.savefig(self.soh_hist_url)



# 自定义单峰正态分布拟合
# 定义正态分布函数
def normal_distribution(x, mean, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
# 定义损失函数（最小二乘法）
def loss_function(params, x, y):
    mean, std = params
    y_pred = normal_distribution(x, mean, std)
    return np.sum((y - y_pred) ** 2)
# 梯度下降算法
def gradient_descent(params, x, y, learning_rate, epochs):
    mean, std = params
    for i in range(epochs):
        y_pred = normal_distribution(x, mean, std)
        # loss函数的对均值与方差的偏导数
        d_loss_d_mean = -2 * np.sum((y - y_pred) * y_pred * (x - mean) / std )
        d_loss_d_std = -2 * np.sum((y - y_pred) * y_pred * ((x - mean) ** 2 - std ** 2) / (std ** 3))
        # 向着偏导数的负方向递减
        mean -= learning_rate * d_loss_d_mean
        std -= learning_rate * d_loss_d_std
        # if abs(d_loss_d_mean) < 1e-5 and abs(d_loss_d_std) < 1e-5:
        # # 01-24 16:33 更新
        # if abs(d_loss_d_mean) < 5e-2 and abs(d_loss_d_std) < 5e-2:
        # # 01-25 11:07 更新
        if abs(d_loss_d_mean) < 1e-3 and abs(d_loss_d_std) < 1e-3:
            log.logger.debug(f'epoch:{i}')
            log.logger.debug(f'd_loss_d_mean:{d_loss_d_mean}')
            log.logger.debug(f'd_loss_d_std:{d_loss_d_std}')
            # print('epoch:', i)
            # print('d_loss_d_mean:', d_loss_d_mean)
            # print('d_loss_d_std:', d_loss_d_std)
            
            loss = loss_function([mean, std], x, y)
            log.logger.debug(f'loss:{loss}')
            # print('loss:', loss)
            break
    return mean, std


"""
    # 01-22 17:46 更新
    -- SOH分布筛选防漏机制
    输入： 球坐标处理后的参数 param_after_sphere
    
    机理： 
    即针对球坐标筛选的参数，画柱状图，根据柱状图最高点对应的参数值，以及球坐标筛选的参数的均值和中位值，在这几个数中间确定最大值和最小值，
    然后选择均值作为SOH
        
"""
def process_param_if_non_result_after_SOH_distribution(param_after_sphere):
    
    col = 'SOH'
    
    # bum_num_base = 10
    # bin_num_multiply = [i+1 for i in range(int(len(param_after_sphere) / bum_num_base / 10))]
    # # bin_num_multiply = [1,2,3,4,5]
    # bin_num_rolls = [bum_num_base * i for i in bin_num_multiply]
    
    bin_num_rolls = [int(len(param_after_sphere)/10), int(len(param_after_sphere)/10)+1]
    
    sort_nums = 2
    
    aim_SOH_values_by_multi_hist_times = []
    for bin_nums in bin_num_rolls:
        aim_SOH_values_by_highest_hist_index = get_value_of_hist(param_after_sphere[col], bin_nums, sort_nums)
        aim_SOH_values_by_multi_hist_times.extend(aim_SOH_values_by_highest_hist_index)
        
    
    mean_value = param_after_sphere[col].mean()
    median_value = param_after_sphere[col].median()
    
    aim_SOH_values_by_multi_hist_times.append(mean_value)
    aim_SOH_values_by_multi_hist_times.append(median_value)
    
    aim_SOH_values_by_multi_hist_times = sorted(aim_SOH_values_by_multi_hist_times)
    
    # max_of_three_value = np.max([aim_SOH_values_by_highest_hist_index, mean_value, median_value])
    # min_of_three_value = np.min([aim_SOH_values_by_highest_hist_index, mean_value, median_value])
    
    max_of_three_value = np.array(aim_SOH_values_by_multi_hist_times).max()
    min_of_three_value = np.array(aim_SOH_values_by_multi_hist_times).min()
    
    log.logger.debug(f'{max_of_three_value}, {min_of_three_value}, {aim_SOH_values_by_multi_hist_times}')
    # print(max_of_three_value, min_of_three_value, aim_SOH_values_by_multi_hist_times)
    
    aim_SOH_value_out = np.mean([max_of_three_value, min_of_three_value])
    
    # if max_of_three_value - min_of_three_value < 1:
    #     aim_SOH_value_out = np.mean([max_of_three_value, min_of_three_value])
    
    # else:
    #     new_array_of_value = np.linspace(min_of_three_value, max_of_three_value, 10*int(max_of_three_value-min_of_three_value))
    #     # print('new_array_of_value', new_array_of_value)
    #     aim_SOH_value_out = get_value_of_hist(new_array_of_value)
    
    return aim_SOH_value_out


def get_value_of_hist(params_to_be_histed, bin_nums, sort_nums):
    
    # hist_height, bins, patches = plt.hist(params_to_be_histed, bins=bin_nums)
    # # 与上面代码一样, 只是不画图
    hist_height, bins = np.histogram(params_to_be_histed, bins=bin_nums)
    
    # highest_hist_index = hist_height.argmax()
    
    # 返回最大的多个值
    highest_hist_index = hist_height.argsort()[::-1][:sort_nums]
    
    aim_SOH_values_by_highest_hist_index = bins[highest_hist_index]
    
    return aim_SOH_values_by_highest_hist_index


def run_soh_distribution(param_after_sphere, save_flag=False, if_plot=False, soh_hist_url='', return_param=False):
    # 01-26 14:47 更新
    # 这个地方没有为param_after_soh_fit赋值？？？？？
    if param_after_sphere['SOH'].max() - param_after_sphere['SOH'].min() < 3:
    # if param_after_sphere['SOH'].max() - param_after_sphere['SOH'].min() < 1:
        # soh_list.append(param_after_sphere['SOH'].mean())
        # soh_fit_std_of_cells_list.append(param_after_sphere['SOH'].std())
        # print('soh_list_by_mean', soh_list)
        
        # out_soh = process_param_if_non_result_after_SOH_distribution(param_after_sphere)
        # # # 前面乘以一个数, 后面再除回来
        # # out_soh = out_soh / multi_num
        # soh_via_hist_peaks_list.append(out_soh)
        # print('out_soh_list_by_hist_peaks', soh_via_hist_peaks_list)
        log.logger.debug(f"soh 最大 差值为 {param_after_sphere['SOH'].max() - param_after_sphere['SOH'].min()}, 不用画分布")
        # print(f"soh 最大 差值为 {param_after_sphere['SOH'].max() - param_after_sphere['SOH'].min()}, 不用画分布")
        soh_this_cell = param_after_sphere['SOH'].mean()
        std_this_cell = param_after_sphere['SOH'].std()

    else:
        multi_num = 1
        while (param_after_sphere['SOH'].max() - param_after_sphere['SOH'].min()) < 20:
            # 01-26 11:25 备注: 这里必须要除回去变成原数据之后, 再重新去乘新的multi_num
            param_after_sphere['SOH'] = param_after_sphere['SOH'] / multi_num
            
            multi_num *= 2
            param_after_sphere['SOH'] = param_after_sphere['SOH'] * multi_num

        log.logger.debug('multi_num', multi_num, '\n', param_after_sphere['SOH'].head())    
        # print('multi_num', multi_num, '\n', param_after_sphere['SOH'].head())
        
        
        learning_rate = 1
        mean_fit = param_after_sphere['SOH'].mean()
        std_fit = param_after_sphere['SOH'].std() 
        # std_fit = 50
        if len(param_after_sphere) > 400:
            bin_num = 40
        else:
            bin_num = int(len(param_after_sphere)/10)
            
        epochs = 10000
        list_of_df_max_density_of_SOH = [[]]
        
        # if_plot = True
        # soh_hist_url = os.path.join(fig_url, vol_col+'.png')
        
        spbsd_obj = Screen_Param_By_SOH_Distribution(mean_fit=mean_fit, std_fit=std_fit, param_df_list=[param_after_sphere], bin_num=bin_num, learning_rate=learning_rate, epochs=epochs, if_plot=if_plot, soh_hist_url=soh_hist_url)
        list_of_df_max_density_of_SOH, std_fit_outcome = spbsd_obj.run_this_class()
        
        
        # 前面乘以一个数, 后面再除回来
        param_after_soh_fit = list_of_df_max_density_of_SOH[0]
        param_after_soh_fit['SOH'] = param_after_soh_fit['SOH'] / multi_num                    
            
        # if save_flag:
        #     save_output_at_different_phases([param_after_soh_fit], vol_col, outpath[1], 2, 1)
            
        
        # soh_list.append(param_after_soh_fit['SOH'].iloc[0])
        # # 前面乘以一个数, 后面再除回来
        # soh_fit_std_of_cells_list.append(std_fit_outcome[0] / multi_num)
        
        # print('soh_list_by_fit', soh_list)


        # out_soh = process_param_if_non_result_after_SOH_distribution(param_after_sphere)
        # # 前面乘以一个数, 后面再除回来
        # out_soh = out_soh / multi_num
        # soh_via_hist_peaks_list.append(out_soh)
        # print('out_soh_list_by_hist_peaks', soh_via_hist_peaks_list)
        
        soh_this_cell = param_after_soh_fit['SOH'].iloc[0]
        std_this_cell = std_fit_outcome[0] / multi_num
    
    # 返回参数的 DataFrame   
    if return_param:
        return param_after_soh_fit, std_this_cell
    # 返回计算好的 SOH
    else:
        return soh_this_cell, std_this_cell
    
