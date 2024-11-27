import logging
import sys
from logging import Logger, StreamHandler, FileHandler, DEBUG
from typing import Union


class LogUtility(Logger):
    __LOG_FORMAT = (
        "[%(asctime)s] %(name)s (%(levelname)s) %(funcName)s:%(lineno)d — %(message)s"
    )

    __DATE_FORMAT = "%y/%m/%d-%H:%M:%S"

    def __init__(
        self,
        name: str,
        log_format: str = __LOG_FORMAT,
        date_format: str = __DATE_FORMAT,
        level: Union[int, str] = DEBUG,
        *args,
        **kwargs
    ) -> None:
        super().__init__(name, level)
        self.formatter = logging.Formatter(log_format, date_format)
        self.addHandler(self.__get_file_handler())
        self.addHandler(self.__get_stream_handler())

    def __get_stream_handler(self) -> StreamHandler:
        handler = StreamHandler(sys.stdout)
        handler.setFormatter(self.formatter)
        return handler

    def __get_file_handler(self) -> FileHandler:
        handler = FileHandler("./app_server.log", encoding="utf-8")
        handler.setFormatter(self.formatter)
        return handler

    @staticmethod
    def create(log_level: str = "DEBUG") -> Logger:
        logging.setLoggerClass(LogUtility)
        logger = logging.getLogger("OSS-RAG")
        logger.setLevel(log_level)
        return logger

    @staticmethod
    def getLogger() -> Logger:
        logger = logging.getLogger("OSS-RAG")
        return logger
