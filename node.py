from logging import Logger

from fastapi import FastAPI
from common.plugin import PluginManager
from common.plugin.helper import (
    VectorStorePluginCore,
    LLMPluginCore,
    ChunkerPluginCore,
    FilterPluginCore,
)
from common.api import APIManager
from demo.demo_manager import DemoManager
from util import LogUtility, AppConfig, FileSystem
import json
import os
import subprocess

# Specify the model path to use tiktoken's tokenizer model (cl100k_base, gpt2, etc.) offline.
os.environ["TIKTOKEN_CACHE_DIR"] = os.getcwd() + "/data"


class Node:
    _logger: Logger
    parameters: dict
    config_data: dict
    _plugin_manager: PluginManager
    _api_manager: APIManager
    _demo_manager: DemoManager

    def __init__(self, parameters: dict):
        self.config_data = AppConfig().get()
        self.parameters = self.__check_parameters(parameters)
        self.procs = []

        self._logger = LogUtility.create(parameters["log_level"])
        self._logger.debug("---- config_data:" + json.dumps(self.config_data))

        self._plugin_manager = PluginManager(
            options=self.parameters, config_data=self.config_data, logger=self._logger
        )

        self._api_manager = APIManager(
            llm=self.get_llm(),
            vector_store=self.get_vector_store(),
            reranker=self.get_reranker(),
            chunker=self.get_chunker(),
            filter=self.get_filter(),
            options=self.config_data["rag"],
            logger=self._logger,
        )

        self._demo_manager = DemoManager()

        self.router = self._api_manager.router

        for plugin_type in self.config_data["plugins"].keys():
            for plugin_name in self.config_data["plugins"][plugin_type].keys():
                plugin = self._plugin_manager.plugins[plugin_type][plugin_name]
                plugin_router = plugin.registerAPIs(self)

                if plugin_router is not None:
                    self.router.include_router(plugin_router)

    async def initialize(self, app: FastAPI):
        await self._demo_manager.startup(app, self.config_data)

        # Print summary with index page link
        print("\n" + "=" * 50)
        print("All services started successfully!")
        port = self.config_data.get("http").get("port")
        dashboard_url = f"http://localhost:{port}/demo"
        print(f"\nAvailable demos: {dashboard_url}")
        print("=" * 50 + "\n")

    async def finalize(self):
        await self._demo_manager.shutdown()

    def get_llm(self) -> LLMPluginCore:
        return self._get_plugin("llm")

    def get_reranker(self):
        return self._get_plugin("reranker")

    def get_vector_store(self) -> VectorStorePluginCore:
        return self._get_plugin("vector_store")

    def get_embedding_model(self):
        return self._get_plugin("embedding_model").getEmbeddings()

    def get_chunker(self) -> ChunkerPluginCore:
        return self._get_plugin("chunker")

    def get_filter(self) -> FilterPluginCore:
        return self._get_plugin("filter")

    def _get_plugin(self, plugin_type: str):
        plugin_config = dict(self.config_data["plugins"]).get(plugin_type)
        if plugin_config is None:
            return None

        name = next(iter(plugin_config.keys()))
        try:
            plugin = self._plugin_manager.plugins[plugin_type][name]
        except:
            plugin = None

        return plugin

    def __check_parameters(self, parameters):
        if parameters["log_level"] is None:
            if "logging" in self.config_data and "level" in self.config_data["logging"]:
                parameters["log_level"] = self.config_data["logging"]["level"]
            else:
                parameters["log_level"] = "DEBUG"

        if parameters["directory"] is None:
            if "plugin_dir" in self.config_data:
                parameters["directory"] = self.config_data["plugin_dir"]
            else:
                parameters["directory"] = FileSystem.get_plugins_directory()

        return parameters
