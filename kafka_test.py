from kafka import KafkaConsumer
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
bootstrap_servers = "chuxing.gandongyun.com:9092,chuxing.gandongyun.com:9292,chuxing.gandongyun.com:9192"
topic = "clean_datasource"
topic_report = "report_json"
group_id = "alg"

def get_two_historical_messages(target_topic):
    consumer = None
    try:
        consumer = KafkaConsumer(
            target_topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=None,  # 匿名消费者，不影响原有 group
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,
        )

        logger.info(f"开始获取 {target_topic} 的2条历史消息...")

        records = []
        while len(records) < 2:
            polled = consumer.poll(timeout_ms=2000, max_records=2 - len(records))
            for partition, recs in polled.items():
                records.extend(recs)
            if not polled:  # 连续 poll 没有新消息
                break

        if not records:
            logger.info("未获取到任何消息")
            return

        for idx, record in enumerate(records[:2], start=1):
            print(f"\n===== 消息 {idx} =====")
            print(f"分区: {record.partition}")
            print(f"偏移量: {record.offset}")
            print(f"时间戳: {record.timestamp}")
            print("内容:")
            print(json.dumps(record.value, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"获取消息失败: {str(e)}", exc_info=True)
    finally:
        if consumer:
            consumer.close()
            logger.info("消费者已关闭")


if __name__ == "__main__":
    # 选择要查看的主题（clean_datasource 或 report_json）
    get_two_historical_messages(topic)
