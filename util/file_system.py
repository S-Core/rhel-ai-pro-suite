import os
from typing import Optional

import yaml


class FileSystem:

    @staticmethod
    def __get_base_dir():
        """At most all application packages are just one level deep"""
        current_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(current_path, "..")

    @staticmethod
    def __get_config_directory() -> str:
        base_dir = FileSystem.__get_base_dir()
        return os.path.join(base_dir, "config")

    @staticmethod
    def get_plugins_directory() -> str:
        base_dir = FileSystem.__get_base_dir()
        return os.path.join(base_dir, "plugins")

    @staticmethod
    def load_configuration(
        name: str = "configuration.yml", config_directory: Optional[str] = None
    ) -> dict:
        if config_directory is None:
            config_directory = FileSystem.__get_config_directory()
        with open(os.path.join(config_directory, name), encoding="utf8") as file:
            input_data = yaml.safe_load(file)
        return input_data