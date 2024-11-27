from logging import Logger
from typing import List, Any, Dict

from .helper.plugin_utility import PluginUtility
from .plugin_loader import (
    PluginLoader,
    GeneralPluginLoader,
    VectorStorePluginLoader,
    ModelPluginLoader,
)

from util import AppConfig


class PluginManager:
    """
    The PluginManager class is responsible for managing plugins in the system.

    Attributes:
        embedding_models (Dict): A dictionary of embedding models loaded from plugins.
        vector_stores (Dict): A dictionary of vector stores loaded from plugins.
        llms (Dict): A dictionary of llms (low-level models) loaded from plugins.
    """

    _logger: Logger
    plugins: Dict

    def __init__(self, **kwargs) -> None:
        self._logger = kwargs["logger"]
        self.plugins_package: str = kwargs["options"]["directory"]
        self.plugin_util = PluginUtility(self._logger)
        config_data = AppConfig().get()
        self.plugins = {}

        self.plugins["embedding_model"] = self._load_plugins(
            ModelPluginLoader(self._logger, self.plugins_package),
            config_data,
            "embedding_model",
        )

        self.plugins["vector_store"] = self._load_plugins(
            VectorStorePluginLoader(
                self._logger,
                self.plugins_package,
                self.plugins["embedding_model"],
            ),
            config_data,
            "vector_store",
        )

        self.plugins["reranker"] = self._load_plugins(
            ModelPluginLoader(self._logger, self.plugins_package),
            config_data,
            "reranker",
        )

        self.plugins["llm"] = self._load_plugins(
            GeneralPluginLoader(self._logger, self.plugins_package),
            config_data,
            "llm",
        )

        self.plugins["evaluation"] = self._load_plugins(
            GeneralPluginLoader(self._logger, self.plugins_package),
            config_data,
            "evaluation",
        )

        self.plugins["chunker"] = self._load_plugins(
            GeneralPluginLoader(self._logger, self.plugins_package),
            config_data,
            "chunker",
        )

        self.plugins["filter"] = self._load_plugins(
            GeneralPluginLoader(self._logger, self.plugins_package),
            config_data,
            "filter",
        )

    def _load_plugins(self, loader: PluginLoader, config_data: dict, plugin_type: str):
        return loader.load_plugins(config_data, plugin_type)
