import logging

logger = logging.getLogger("logger")
# 设置日志级别（例如，DEBUG、INFO、WARNING、ERROR、CRITICAL）
logger.setLevel(logging.DEBUG)

# 创建一个处理程序，例如，将日志记录到文件
file_handler = logging.FileHandler('./logs/example.log')

# 创建一个格式化程序，定义日志消息的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加处理程序到 logger
logger.addHandler(file_handler)
