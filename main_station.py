import os
import pandas as pd
# from tool.pvlog import Logger
from tool_utils.pvlog import Logger

from datetime import datetime
import shutil
import traceback

from pyCxx.run_alg import alg_execution, check_folder

from kafka import KafkaConsumer, TopicPartition
import configparser
import json
from tool_utils.kafka_to_monitor import send_report_msg
from tool_utils.generate_report import generate_report
from tool_utils.data_megre import DataMerge

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client


level = 'info'
log = Logger('./logs/main_out.log', level=level)

def unpack_flatten(zip_path, extract_to):
    extract_to = Path(extract_to)
    extract_to.mkdir(exist_ok=True)
    with ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.endswith('/'):
                target = Path(member).name
                zf.extract(member, extract_to)
                (extract_to / member).rename(extract_to / target)

'''
@ 检查csv_origin是否存在待处理csv， 若存在则读取并处理生成为报告， 并在读取完成后，将当前csv文件删除，并存储到csv_history中 
'''

def run_generate_report_zip(log):
    df_raw = []
    file_raw = []
    origin_data_folder = "csv_origin"
    history_data_folder = "csv_history"
    # 创建用于放置历史文件的目录
    history_folder_path = os.path.join('./csv_history', datetime.now().strftime("%Y%m%d"))
    os.makedirs(history_folder_path, exist_ok=True)

    # 循环处理zip文件 
    for filename in os.listdir(origin_data_folder):
        # 解压zip文件到同名的文件夹中
        if filename.endswith('.zip'):
            origin_zip_path = os.path.join(origin_data_folder, filename)
            file_name_without_extension = filename.split('.')[0]
            origin_path = os.path.join(origin_data_folder, file_name_without_extension)
            history_path = os.path.join(history_folder_path, file_name_without_extension)
            error_path = os.path.join('csv_error', file_name_without_extension)

            if not os.path.exists(origin_path):
                os.makedirs(origin_path)
            try:
                # 1. 解压zip文件夹，解压后删除原zip文件
                # shutil.unpack_flatten(origin_path, extract_dir)
                with ZipFile(origin_zip_path) as zf:
                    members = zf.namelist()
                    has_nested = any('/' in m for m in members)
                    if has_nested:
                        unpack_flatten(origin_zip_path, origin_path)
                    else:
                        shutil.unpack_archive(origin_zip_path, origin_path)
                # 删除原始zip文件
                # os.remove(origin_zip_path)
                # 2. 读取解压后的文件夹中的json文件，并从中读取相关的配置参数
                config_json_path = os.path.join(origin_path, "config.json")
                config_data = None
                if os.path.exists(config_json_path):
                    try:
                        with open(config_json_path, "r", encoding="utf-8") as f:
                            config_data = json.load(f)
                    except Exception as e:
                        log.logger.error(f"读取配置文件失败: {traceback.print_exc()}")
                # 3. 读取原始csv文件--pulse文件测试数据，读取完成后，进行算法计算处理
                merge_data = DataMerge(origin_path)
                df_pulse_data  = merge_data.out_all_pulse_data()
                # 4. 读取原始csv文件--autocap文件，并进行算法处理
                dict_autocap_data = merge_data.out_all_capacity_data()
                # 5. 读取原始csv文件--balance文件，并进行算法处理
                dict_balance_data  = merge_data.out_all_balance_data()
            except Exception as e:
                shutil.move(origin_path, error_path)
                log.logger.error("读取原始文件%s失败，请检查文件格式是否正确!"%(file_name_without_extension))
                return

            # 执行算法模块
            try:
                # 6. 调用算法执行模块进行计算
                alg_rlt = alg_execution(df_pulse_data, dict_autocap_data, dict_balance_data, config_data, file_name_without_extension)
                log.logger.info("算法执行完成，返回%s"%(alg_rlt['ErrorCode'][0]))
            except Exception as e:
                # shutil.move(origin_path, error_path)
                log.logger.error("文件%s执行算法计算报错，请检查文件格式是否正确!"%(file_name_without_extension))
                return
            
            if alg_rlt['ErrorCode'][0] == 0:
                log.logger.info("文件%s生成报告成功"%(config_data))
                # shutil.move(origin_path, history_path) 
            else:
                log.logger.error("文件%s生成报告失败，原因：%s"%(config_data, alg_rlt['ErrorCode'][2]))
                # shutil.move(origin_path, error_path)

        
        # # 解压zip文件到同名的文件夹中
        # if filename.endswith('.zip'):
        #     origin_dir = os.path.join(origin_data_folder, filename)
        #     file_name_without_extension = filename.split('.')[0]
        #     extract_dir = os.path.join(origin_data_folder, file_name_without_extension)
        #     if not os.path.exists(extract_dir):
        #         os.makedirs(extract_dir)
        #     try:
        #         # 1. 解压zip文件夹，解压后删除原zip文件
        #         shutil.unpack_archive(origin_dir, extract_dir)
        #         # 2. 读取解压后的文件夹中的json文件，并从中读取相关的配置参数
        #         config_json_path = os.path.join(extract_dir, "config.json")
        #         config_data = None
        #         if os.path.exists(config_json_path):
        #             try:
        #                 with open(config_json_path, "r", encoding="utf-8") as f:
        #                     config_data = json.load(f)
        #             except Exception as e:
        #                 log.logger.error(f"读取配置文件失败: { traceback.print_exc()}")
        #                 continue
        #         # 3. 读取原始csv文件--pulse文件测试数据，读取完成后，进行算法计算处理
        #         merge_data = DataMerge(extract_dir)
        #         df_pulse_data  = merge_data.out_all_pulse_data()
        #         # 4. 读取原始csv文件--autocap文件，并进行算法处理
        #         dict_autocap_data = merge_data.out_all_capacity_data()
        #         # df_balance_data = pd.read_csv(os.path.join(extract_dir, "balance.csv"))
        #         # 5. 读取原始csv文件--balance文件，并进行算法处理
        #         dict_balance_data  = merge_data.out_all_balance_data()
        #         # 6. 调用算法执行模块进行计算
        #         alg_rlt = alg_execution(df_pulse_data, dict_autocap_data, dict_balance_data, config_data, file_name_without_extension)
        #         log.logger.info("算法执行完成，返回%s"%(alg_rlt['ErrorCode'][0]))
            #     if alg_rlt['ErrorCode'][0] == 0:
            #         log.logger.info("文件%s生成报告成功"%(config_data))
            #         # shutil.move(origin_path, history_path) 
            #     else:
            #         log.logger.error("文件%s生成报告失败，原因：%s"%(config_data, alg_rlt['ErrorCode'][2]))
            #         # shutil.move(origin_path, error_path)

            # except Exception as e:
            #     log.logger.error(f"解压zip文件失败: { traceback.print_exc()}")
            #     log.logger.error("解压zip文件失败，请检查文件格式是否正确!")
            #     continue


from zipfile import ZipFile
from pathlib import Path
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
        origin_zip_path = os.path.join(origin_data_folder, filename)

        # origin_dir = os.path.join(origin_data_folder, filename)
        file_name_without_extension = filename.split('.')[0]
        origin_path = os.path.join(origin_data_folder, file_name_without_extension)
        history_path = os.path.join(history_folder_path, file_name_without_extension)
        error_path = os.path.join('csv_error', file_name_without_extension)

        if not os.path.exists(origin_path):
            os.makedirs(origin_path)
        try:
            # 1. 解压zip文件夹，解压后删除原zip文件
            # shutil.unpack_flatten(origin_path, extract_dir)
            with ZipFile(origin_zip_path) as zf:
                members = zf.namelist()
                has_nested = any('/' in m for m in members)
                if has_nested:
                    unpack_flatten(origin_zip_path, origin_path)
                else:
                    shutil.unpack_archive(origin_zip_path, origin_path)
            # 删除原始zip文件
            os.remove(origin_zip_path)
            # 2. 读取解压后的文件夹中的json文件，并从中读取相关的配置参数
            config_json_path = os.path.join(origin_path, "config.json")
            config_data = None
            if os.path.exists(config_json_path):
                try:
                    with open(config_json_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                except Exception as e:
                    log.logger.error(f"读取配置文件失败: {traceback.print_exc()}")
            # 3. 读取原始csv文件--pulse文件测试数据，读取完成后，进行算法计算处理
            merge_data = DataMerge(origin_path)
            df_pulse_data  = merge_data.out_all_pulse_data()
            # 4. 读取原始csv文件--autocap文件，并进行算法处理
            dict_autocap_data = merge_data.out_all_capacity_data()
            # 5. 读取原始csv文件--balance文件，并进行算法处理
            dict_balance_data  = merge_data.out_all_balance_data()
        except Exception as e:
            shutil.move(origin_path, error_path)
            log.logger.error("读取原始文件%s失败，请检查文件格式是否正确!"%(file_name_without_extension))
            send_report_msg(bootstrap_servers, topic_report, current_date, file_name_without_extension, '读取原始文件失败，请检查文件格式是否正确!',-1)
            return
        # 执行算法模块
        try:
            # 6. 调用算法执行模块进行计算
            alg_rlt = alg_execution(df_pulse_data, dict_autocap_data, dict_balance_data, config_data, file_name_without_extension)
            log.logger.info("算法执行完成，返回%s"%(alg_rlt['ErrorCode'][0]))
        except Exception as e:
            shutil.move(origin_path, error_path)
            log.logger.error("文件%s执行算法计算报错，请检查文件格式是否正确!"%(file_name_without_extension))
            send_report_msg(bootstrap_servers, topic_report, current_date, file_name_without_extension, '文件执行算法计算报错，请检查文件格式是否正确!',-1)
            return
        
        if alg_rlt['ErrorCode'][0] == 0: 
            # 发送报告消息
            log.logger.info("文件%s生成报告成功"%(file_name_without_extension))
            send_report_msg(bootstrap_servers, topic_report, current_date, file_name_without_extension, '成功', alg_rlt['ErrorCode'][0])
            shutil.move(origin_path, history_path)
        else:
            log.logger.error("文件%s生成报告失败，原因：%s"%(file_name_without_extension, alg_rlt['ErrorCode'][2]))
            send_report_msg(bootstrap_servers, topic_report, current_date, file_name_without_extension, alg_rlt['ErrorCode'][2], alg_rlt['ErrorCode'][0])
            shutil.move(origin_path, error_path)
        # df_raw.clear()
        # file_raw.clear()    
    else:
        log.logger.error(f"{origin_data_folder}/{filename} 文件不存在")

run_mode = 2  # 1：正式版本（kafka消费），2：kafka消费测试版本（消费历史数据）3：离线测试版本（读取本地zip包）
# 默认负电流为充电，正电流为放电
if __name__ == "__main__":
    check_folder()

    #################离线测试版本（读取本地zip包）#################################
    if run_mode == 3:
        run_generate_report_zip(log)  # 算法手动测试用代码
    
    if run_mode == 1 or run_mode == 2:
        # 创建配置文件对象
        file = 'kafkaconfig.ini'
        conf = configparser.ConfigParser()
        # 读取配置文件
        conf.read(file, encoding='utf-8')
        bootstrap_servers = conf['kafka_test']['bootstrap_servers']
        group_id = conf['kafka_test']['group_id']
        topic_clean = conf['kafka_test']['topic']
        topic_report = conf['kafka_test']['topic_report']
        reset_mode = conf['kafka_test']['reset_mode']

        # 初始化 cos 客户端
        cos_config =  conf['cos']
        secret_id = cos_config['secret_id']
        secret_key = cos_config['secret_key']
        region = cos_config['region']
        bucket = cos_config['bucket']
        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
        client = CosS3Client(config)

        log.logger.info("程序运行版本号：ver1.0_bulid_20250813")
        log.logger.info("当前consumer配置：%s %s %s"%(bootstrap_servers, group_id, topic_clean))

    if run_mode == 1:
    #################正式版本，读取未消费消息#################################
        # 获取csv文件路径
        consumer = KafkaConsumer(
                            topic_clean,
                            bootstrap_servers=bootstrap_servers,
                            auto_offset_reset=reset_mode,
                            enable_auto_commit=False,
                            group_id=group_id,
                        )
        log.logger.info("当前订阅话题：%s"%(consumer.topics()))

        # 获取未消费消息，从最早的开始消费
        for msg in consumer:
            log.logger.info(f"收到消息主题：{msg.topic}")
            if msg.topic == topic_clean:
                # 反序列化消息
                msg_value = json.loads(msg.value.decode())
                log.logger.info(f"收到消息：{msg_value}")
                # 手动提交偏移量
                consumer.commit()
                # 报告生成
                run_generate_report_from_kafka(log, bootstrap_servers, topic_report, msg_value, client, bucket)

    ########################测试版本，读取已消费消息###############################
    if run_mode == 2:
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=False,
            group_id=None,  # 无状态消费
            consumer_timeout_ms=5000
        )
        consumer.subscribe([topic_clean])
        consumer.poll(timeout_ms=1000)  # 触发分区分配

        partitions = consumer.partitions_for_topic(topic_clean)
        if partitions:
            topic_partitions = [TopicPartition(topic_clean, p) for p in partitions]
            end_offsets = consumer.end_offsets(topic_partitions)

            for tp in topic_partitions:
                if end_offsets[tp] > 0:
                    consumer.seek(tp, end_offsets[tp] - 1)

            for msg in consumer:
                print(f"最新消息: {msg.value.decode('utf-8')}")
                msg_value = json.loads(msg.value.decode())
                # log.info(f"分区 {partition.partition} 已消费的最新消息：{last_message.value}")
                run_generate_report_from_kafka(log, bootstrap_servers, topic_report, msg_value, client, bucket)
                # break  # 只取一条
        else:
            print("主题不存在或无分区")
        
