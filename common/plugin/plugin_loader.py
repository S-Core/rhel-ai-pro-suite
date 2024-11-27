from abc import abstractmethod
import os
from importlib import import_module
from logging import Logger
from typing import List, Any, Dict

from .helper.plugin_core import IPluginRegistry, PluginCore
from .helper.plugin_utility import PluginUtility

import subprocess
import sys


class PluginLoader:
    """
    The PluginLoader class is responsible for loading and managing plugins.

    Attributes:
        plugins (Dict): A dictionary to store the loaded plugins.

    Methods:

        load_plugins(self, config_data: dict, package_name: str):
            Loads the plugins based on the provided configuration data.

    """

    plugins: Dict

    def __init__(self, logger: Logger, plugins_package: str) -> None:
        """
        Initializes the PluginLoader instance.

        Args:
            logger (Logger): The logger instance for logging.
            plugins_package (str): The package name where the plugins are located.

        """
        self._logger = logger
        self.plugins_package: str = plugins_package
        self.plugin_util = PluginUtility(self._logger)
        self.plugins = dict()

    def load_plugins(self, config_data: dict, package_name: str):
        """
        Loads the plugins based on the provided configuration data.

        Args:
            config_data (dict): The configuration data containing the plugin information.
            package_name (str): The name of the package containing the plugins.

        Returns:
            dict: A dictionary containing the loaded plugins.

        """
        config = dict(config_data["plugins"]).get(package_name)
        if config is None:
            return

        self.plugins.clear()
        IPluginRegistry.plugin_registries.clear()

        self._logger.debug(f"Searching for plugins under package {config.keys()}")

        plugins_path = list(config.keys())
        plugins_package = os.path.basename(os.path.normpath(self.plugins_package))
        self.__search_for_plugins_in(
            config, plugins_path, package_name, plugins_package
        )

        return self.plugins

    def __search_for_plugins_in(
        self,
        config: dict,
        plugins_path: List[str],
        package_name: str,
        plugins_package: str,
    ):
        """
        Searches for plugins in the specified directories.

        Args:
            config (dict): The configuration data containing the plugin information.
            plugins_path (List[str]): The list of directories to search for plugins.
            package_name (str): The name of the package containing the plugins.
            plugins_package (str): The name of the plugins package.

        """
        for directory in plugins_path:
            entry_point = self.plugin_util.setup_plugin_configuration(
                package_name, directory
            )
            if entry_point is not None:
                plugin_name, _ = os.path.splitext(entry_point)
                # Importing the module will cause IPluginRegistry to invoke its __init__ function
                import_target_module = f".{package_name}.{directory}.{plugin_name}"
                module = import_module(import_target_module, plugins_package)
                self._check_loaded_plugin_state(config, directory, module)
            else:
                self._logger.debug(f"No valid plugin found in {package_name}")

    def _check_loaded_plugin_state(self, config: dict, key: str, plugin_module: Any):
        """
        Checks the state of the loaded plugin.

        Args:
            config (dict): The configuration data containing the plugin information.
            key (str): The key of the plugin in the configuration data.
            plugin_module (Any): The loaded plugin module.

        """
        if len(IPluginRegistry.plugin_registries) > 0:
            latest_module = IPluginRegistry.plugin_registries[-1]
            latest_module_name = latest_module.__module__
            current_module_name = plugin_module.__name__
            if current_module_name == latest_module_name:
                self._logger.debug(
                    f"Successfully imported module `{current_module_name}`"
                )
                # Initialize the plugin object and register it in the plugins dictionary
                self.plugins[key] = self._initialize_plugin(latest_module, config[key])
            else:
                self._logger.error(
                    f"Expected to import -> `{current_module_name}` but got -> `{latest_module_name}`"
                )
            # Clear plugins from the registry when we're done with them
            IPluginRegistry.plugin_registries.clear()
        else:
            self._logger.error(
                f"No plugin found in registry for module: {plugin_module}"
            )

    @abstractmethod
    def _initialize_plugin(self, latest_module: Any, config: dict):
        """
        Initializes the plugin object.

        Args:
            latest_module (Any): The latest loaded plugin module.
            config (dict): The configuration data for the plugin.

        """
        pass


class GeneralPluginLoader(PluginLoader):
    """
     A plugin loader for general plugins.

     This class extends the base PluginLoader.

    Args:
        logger (Logger): The logger instance to use for logging.
        plugins_package (str): The name of the package containing the plugins.

    """

    def __init__(self, logger: Logger, plugins_package: str) -> None:
        super().__init__(logger, plugins_package)

    def _initialize_plugin(self, latest_module: Any, config: dict):
        plugin = latest_module(self._logger)
        return plugin.invoke(config)


class ModelPluginLoader(PluginLoader):
    """
    A plugin loader for embedding model plugins.

    This class extends the base PluginLoader class and provides functionality
    for loading and initializing embedding model plugins.

    Args:
        logger (Logger): The logger instance for logging messages.
        plugins_package (str): The package name where the plugins are located.

    """

    def __init__(self, logger: Logger, plugins_package: str) -> None:
        super().__init__(logger, plugins_package)

    def __load_model(self, config: dict):
        model_path = config["model_path"]
        config["origin_model_path"] = model_path

        if not os.path.isabs(model_path):
            _, model_name = model_path.split("/")
            models_dir = "models"
            model_dir = os.path.join(models_dir, model_name)

            if not os.path.exists(model_dir):
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)

                print(
                    f"""
-----------------------------------------------
Download the '{model_name}' model from Huggingface.
This may take a long time...
-----------------------------------------------"""
                )
                subprocess.run(["git", "lfs", "install"])
                clone_command = (
                    f"git lfs clone https://huggingface.co/{model_path} {model_dir}"
                )
                subprocess.check_call(clone_command.split())

            if not os.path.exists(model_dir):
                raise Exception(f"Model directory does not exist: {model_dir}")

            config["model_path"] = model_dir

    def _initialize_plugin(self, latest_module: Any, config: dict):
        """
        Initialize the plugin with the given configuration.

        Args:
            latest_module (Any): The latest module of the plugin.
            config (dict): The configuration dictionary.

        Returns:
            Any: The result of invoking the plugin with the configuration.

        """
        plugin = latest_module(self._logger)
        self.__load_model(config)
        return plugin.invoke(config)


class VectorStorePluginLoader(PluginLoader):
    """
    A plugin loader for vector store plugins.

    This class extends the base PluginLoader class and provides functionality
    specific to loading vector store plugins.

    Args:
        logger (Logger): The logger instance to use for logging.
        plugins_package (str): The name of the package containing the plugins.
        embedding_models (dict): A dictionary of embedding models.

    Attributes:
        embedding_models (dict): A dictionary of embedding models.

    """

    def __init__(
        self, logger: Logger, plugins_package: str, embedding_models: dict
    ) -> None:
        super().__init__(logger, plugins_package)
        self.embedding_models = embedding_models

    def _initialize_plugin(self, latest_module: PluginCore, config: dict):
        plugin = latest_module(self._logger)
        return plugin.invoke(
            config, self.embedding_models[config["embedding_model"]].getEmbeddings()
        )
