import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

current_cwd = os.getcwd()    
# 假设vin csv文件存储在某个目录下
csv_dir_path = '/root/data/sp/test_sp/data/csv/vin2025020602_report.xlsx'

def  get_data_from_csv(csv_dir_path):
    """
    从CSV文件中读取数据，并返回为df
    """
    # df_raw = []
    df_raw = pd.read_excel(csv_dir_path)
    filtered_df = df_raw[df_raw['RESULT'] == '成功']
    print(f"Filtered df: {filtered_df}")
    # 处理  filtered_df 中的数据data 
    if 'JSON' in filtered_df.columns:     
        fueltype_list = []
        totalweight_list = []
        productiondate_list = []
        fronttrack_list = []
        len_list = []
        price_list = []
        bodytype_list = []
        brand_list = []
        height_list = []
        maxpower_list = []
        weight_list = []
        name_list = []
        drivemode_list = []
        yeartype_list = []
        json_column = filtered_df['JSON']
        for json_string in json_column:
            try:
                json_data = json.loads(json_string)
                data = json_data.get('data', '')
                result_data =  data["result"]
                fueltype_list.append(result_data['fueltype']) 
                # totalweight_list.append(result_data['totalweight']) 
                productiondate_list.append(result_data['productiondate'])
                # fronttrack_list.append(result_data['fronttrack'])
                len_list.append(result_data['len'])
                price_list.append(result_data['price'])
                # bodytype_list.append(result_data['bodytype'])
                brand_list.append(result_data['brand'])    
                # height_list.append(result_data['height'])    
                maxpower_list.append(result_data['maxpower'])
                weight_list.append(result_data['weight'])
                name_list.append(result_data['name'])
                drivemode_list.append(result_data['drivemode'])
                yeartype_list.append(result_data['yeartype'])
                
            except Exception as e:
                print(f"Error parsing JSON in row : {e}")
                # result_list.append('')
        # filtered_df['totalweight'] = totalweight_list
        filtered_df['productiondate'] = productiondate_list
        filtered_df['fueltype'] = fueltype_list
        # filtered_df['bodytype'] = bodytype_list
        filtered_df['price'] = price_list
        # filtered_df['height'] = height_list
        filtered_df['len'] = len_list
        # filtered_df['fronttrack'] = fronttrack_list
        filtered_df['brand'] = brand_list
        filtered_df['maxpower'] = maxpower_list
        filtered_df['weight'] = weight_list
        filtered_df['name'] = name_list
        filtered_df['drivemode'] = drivemode_list
        filtered_df['yeartype'] = yeartype_list
    return filtered_df
   

if __name__ == "__main__":
   get_data_from_csv(csv_dir_path)