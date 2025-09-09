from kafka import KafkaProducer, KafkaConsumer
import json


class NewKafkaParams(object):
    def __init__(self, kafka_host_port):
        self.kafka_host_port = kafka_host_port

    def get_producer(self):
        producer = KafkaProducer(bootstrap_servers=self.kafka_host_port,
                                 value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'), retries=3,
                                 batch_size=16384,
                                 send_buffer_bytes=131072,
                                 api_version=(0, 10, 2))
        return producer

    def get_consumer(self, group_id):
        consumer = KafkaConsumer(
            bootstrap_servers=self.kafka_host_port, group_id=group_id,
            auto_offset_reset='latest',
            enable_auto_commit=True)
        return consumer
