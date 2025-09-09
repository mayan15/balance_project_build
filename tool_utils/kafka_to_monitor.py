import configparser
import json
from kafka import KafkaProducer
from kafka import KafkaConsumer
from datetime import datetime
from tool_utils.kafka_params import NewKafkaParams
import os
import traceback

from tool_utils.pvlog import Logger


level = 'info'
log = Logger('./logs/kafka.log', level=level)

def send_report_msg(bootstrap_servers, topic, current_time, file_date, error_msg, code_id= 0):
    # 配置 Kafka 的主机地址
    kafka_params = NewKafkaParams(kafka_host_port=bootstrap_servers)

    # 获取生产者对象
    producer = kafka_params.get_producer()

    # 构造报告路径
    project_root = os.getcwd()
    file_path = os.path.join(project_root, 'report', current_time, f"{file_date.split('.')[0]}", 'data.json')

    # 判断文件是否存在
    startChargeSeq = file_date.split('.')[0][5:]
    if not os.path.exists(file_path):
        log.logger.info(f"报告不存在：{startChargeSeq}" )
        report = None
    else:
        # 读取json报告
        with open(file_path, 'r', encoding='utf-8') as file:
            report = json.load(file).get('data')

    # 构造发送消息
    now = datetime.now()
    now.strftime("%Y-%m-%d %H:%M:%S")
    startChargeSeq = file_date.split('.')[0][5:]
    data = {
        "code": code_id,
        "reportJson": report,
        "startChargeSeq": startChargeSeq,
        "msg": error_msg,
        "datetime": str(now)
    }

    # 发送消息到指定的 topic
    try:
        future = producer.send(topic, value=data)
        result = future.get(timeout=60)
        producer.flush()  # 确保消息已经发送
        log.logger.info(f"推送 kafka消息 流水号为：{startChargeSeq}" )
    except Exception as e:
        log.logger.error(f"kafka消息报错 流水号为:{startChargeSeq} 报错信息为：{traceback.print_exc()}" )

    finally:
        producer.close()  # 关闭生产者连接
