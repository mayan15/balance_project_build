#coding: utf-8

import logging
from logging import handlers

'''
日志级别：
CRITICAL 50
ERROR 40
WARNING 30
INFO 20
DEBUG 10
'''

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=30,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别

        # 往文件里写入
        # 指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        #interval: 滚动周期，单位有when指定，比如：when=’D’,interval=1，表示每天产生一个日志文件；
        #timed_rotating_file_handler = handlers.TimedRotatingFileHandler(
        #    filename=filename, when=when, backupCount=backCount,interval=1,encoding='utf-8')
        
        file_rotating_file_handler = handlers.RotatingFileHandler(filename=filename, maxBytes=2000000, backupCount=100,encoding='utf-8')
        #timed_rotating_file_handler.setFormatter(format_str)  # 设置文件里写入的格式
        
        file_rotating_file_handler.setFormatter(format_str)
        
        #self.logger.addHandler(timed_rotating_file_handler)

        self.logger.addHandler(file_rotating_file_handler)
        # 往屏幕上输出
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(format_str)
        self.logger.addHandler(stream_handler)


#log = Logger('all.log', level='info')
#log.logger.info('[测试log] hello, world')
        