import socket
import sys

from util import FileSystem


class AppConfig:
    __config: dict = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            print("__new__ is called\n")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        cls = type(self)
        if not hasattr(cls, "_init"):
            self.__config = self.__validation(FileSystem.load_configuration())
            cls._init = True

    def get(self):
        return self.__config

    def __check_port_is_open(self, host: str = "0.0.0.0", port: int = 8080) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        return result

    def __validation(self, config_data: dict) -> dict:

        if "http" in config_data:
            result = self.__check_port_is_open(
                config_data["http"]["host"], config_data["http"]["port"]
            )

            if result == 0:
                print("Server port is already open")
                sys.exit()

        if "vector_store" not in config_data:
            # type이 elasticsearch 일때
            # milvus 일때
            # weaviate 일때
            pass

        if "embedding_model" not in config_data:
            # type이 local 일때
            # remote 일때
            pass

        if "llm" not in config_data:
            # type이 local 일때
            # remote 일때
            pass

        return config_data
