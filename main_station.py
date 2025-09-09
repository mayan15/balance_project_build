import os
import pandas as pd
# from tool.pvlog import Logger
from tool_utils.pvlog import Logger

from datetime import datetime
import shutil
import traceback

from pyCxx.run_alg import alg_execution, check_folder

from kafka import KafkaConsumer
import configparser
import json
from tool_utils.kafka_to_monitor import send_report_msg
from tool_utils.generate_report import generate_report


from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client


level = 'info'
log = Logger('./logs/main_out.log', level=level)


'''
@ 检查csv_origin是否存在待处理csv， 若存在则读取并处理生成为报告， 并在读取完成后，将当前csv文件删除，并存储到csv_history中 
'''

def run_generate_report(log):
    df_raw = []
    file_raw = []
    origin_data_folder = "csv_origin"
    history_data_folder = "csv_history"
    # 创建用于放置历史文件的
    history_folder_path = os.path.join('./csv_history', datetime.now().strftime("%Y%m%d"))
    os.makedirs(history_folder_path, exist_ok=True)
    for filename in os.listdir(origin_data_folder):
        if filename.endswith('.csv'):
            origin_path = os.path.join(origin_data_folder, filename)
            history_path = os.path.join(history_folder_path, filename)
            error_path = os.path.join('csv_error', filename)
            file_list_id = filename.split('.')
            file_raw.append(file_list_id[0])
            try:
                df_raw.append(pd.read_csv(origin_path))
            except Exception as e:
                shutil.move(origin_path, error_path)
                log.logger.error(f"read csv fails: { traceback.print_exc()}")
                log.logger.error("读取原始文件%s失败，请检查文件格式是否正确!"%(file_list_id[0]))
                continue
            alg_rlt = alg_execution(df_raw, file_raw, log)
            log.logger.info("算法执行完成，返回%s"%(alg_rlt['ErrorCode'][0]))
            if alg_rlt['ErrorCode'][0] == 0:
                log.logger.info("文件%s生成报告成功"%(file_list_id[0]))
                # shutil.move(origin_path, history_path) 
            else:
                log.logger.error("文件%s生成报告失败，原因：%s"%(file_list_id[0], alg_rlt['ErrorCode'][2]))
                # shutil.move(origin_path, error_path)
            df_raw.clear()
            file_raw.clear()
            
'''
@ 触发一次kafka，就生成一次报告，并通知监测平台
''' 
def run_generate_report_from_kafka(log, bootstrap_servers, topic_report, msg_value, client, bucket):
    df_raw = []
    file_raw = []
    origin_data_folder = "csv_origin"
    history_data_folder = "csv_history"
    # 创建用于放置历史文件的
    history_folder_path = os.path.join('./csv_history', datetime.now().strftime("%Y%m%d"))
    os.makedirs(history_folder_path, exist_ok=True)
    # 获取报告excel模板文件路径
    report_data_folder = "report"  
    report_template = os.path.join(report_data_folder, 'report.xlsx')

    current_date = datetime.now().strftime("%Y%m%d")    
    filename = msg_value['fileName']

    ### 从 cos 对象存储中下载到本地
    key = f"{msg_value['filePathClean']}/{filename}"
    local_path = os.path.join(origin_data_folder, filename)

    try:
        # 下载文件到本地
        response = client.get_object(
            Bucket=bucket,
            Key=key,
        )
        with open(local_path, 'wb') as f:
            f.write(response['Body'].get_raw_stream().read())
    except Exception as e:
        print(f"下载失败: {e}")

    if os.path.exists(os.path.join(origin_data_folder, filename)):
        origin_path = os.path.join(origin_data_folder, filename)
        history_path = os.path.join(history_folder_path, filename)
        error_path = os.path.join('csv_error', filename)
        # file_list_id = filename.split('_')
        file_list_id = filename.split('.')
        file_raw.append(file_list_id[0])
        try:
            df_raw.append(pd.read_csv(origin_path))
        except Exception as e:
            shutil.move(origin_path, error_path)
            log.logger.error("读取原始文件%s失败，请检查文件格式是否正确!"%(file_list_id[0]))
            send_report_msg(bootstrap_servers, topic_report, current_date, file_list_id[0], '读取原始文件失败，请检查文件格式是否正确!',-1)
            return
        # 执行算法模块
        try:
            alg_rlt = alg_execution(df_raw, file_raw, log)
        except Exception as e:
            shutil.move(origin_path, error_path)
            log.logger.error("文件%s执行算法计算报错，请检查文件格式是否正确!"%(file_list_id[0]))
            send_report_msg(bootstrap_servers, topic_report, current_date, file_list_id[0], '读取原始文件失败，请检查文件格式是否正确!',-1)
            return
        
        if alg_rlt['ErrorCode'][0] == 0: 
             # 构建excel报告文件夹路径
            json_path = os.path.join(report_data_folder, current_date, file_list_id[0], 'data.json')
            excel_path = os.path.join(report_data_folder, current_date, file_list_id[0], 'report.xlsx')
            # 生成excel报告
            rlt_excel = generate_report(json_path, excel_path, report_template)
            if rlt_excel['ErrorCode'][0] == 0:
                log.logger.info("文件%s生成excel报告成功"%(file_list_id[0]))
            else:
                log.logger.error("文件%s生成excel报告失败，原因：%s"%(file_list_id[0], rlt_excel['ErrorCode'][2]))
            # 发送报告消息
            log.logger.info("文件%s生成报告成功"%(file_list_id[0]))
            send_report_msg(bootstrap_servers, topic_report, current_date, file_list_id[0], '成功', alg_rlt['ErrorCode'][0])
            shutil.move(origin_path, history_path)
        else:
            log.logger.error("文件%s生成报告失败，原因：%s"%(file_list_id[0], alg_rlt['ErrorCode'][2]))
            send_report_msg(bootstrap_servers, topic_report, current_date, file_list_id[0], alg_rlt['ErrorCode'][2], alg_rlt['ErrorCode'][0])
            shutil.move(origin_path, error_path)
        df_raw.clear()
        file_raw.clear()    
    else:
        log.logger.error(f"{origin_data_folder}/{filename} 文件不存在")

# 默认负电流为充电，正电流为放电
if __name__ == "__main__":
    check_folder()
    run_generate_report(log)  # 算法手动测试用代码
    
    # # 创建配置文件对象
    # file = 'kafkaconfig.ini'
    # conf = configparser.ConfigParser()
    # # 读取配置文件
    # conf.read(file, encoding='utf-8')
    # bootstrap_servers = conf['kafka']['bootstrap_servers']
    # group_id = conf['kafka']['group_id']
    # topic_clean = conf['kafka']['topic']
    # topic_report = conf['kafka']['topic_report']
    # reset_mode = conf['kafka']['reset_mode']

    # # 初始化 cos 客户端
    # cos_config =  conf['cos']
    # secret_id = cos_config['secret_id']
    # secret_key = cos_config['secret_key']
    # region = cos_config['region']
    # bucket = cos_config['bucket']
    # config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
    # client = CosS3Client(config)

    # log.logger.info("程序运行版本号：ver1.0_bulid_20250813")
    # log.logger.info("当前consumer配置：%s %s %s"%(bootstrap_servers, group_id, topic_clean))


    # # SP: 20250809 测试kafka连接
    # # from kafka import KafkaAdminClient
    # # admin = KafkaAdminClient(
    # #     bootstrap_servers=bootstrap_servers,
    # #     client_id='test-client'
    # # )
    # # print(admin.list_topics())  # 如果这里就挂了，就是连接问题

    # # 获取csv文件路径
    # consumer = KafkaConsumer(
    #                     topic_clean,
    #                     bootstrap_servers=bootstrap_servers,
    #                     auto_offset_reset=reset_mode,
    #                     enable_auto_commit=False,
    #                     group_id=group_id,
    #                 )
    # log.logger.info("当前订阅话题：%s"%(consumer.topics()))

    # for msg in consumer:
    #     log.logger.info(f"收到消息主题：{msg.topic}")
    #     if msg.topic == topic_clean:
    #         # 反序列化消息
    #         msg_value = json.loads(msg.value.decode())
    #         log.logger.info(f"收到消息：{msg_value}")
    #         # 手动提交偏移量
    #         consumer.commit()
    #         # 报告生成
    #         run_generate_report_from_kafka(log, bootstrap_servers, topic_report, msg_value, client, bucket)