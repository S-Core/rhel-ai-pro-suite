from logging import Logger
from pathlib import Path

from common.plugin.helper import PluginCore
from common.plugin.model import Meta

from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_huggingface import HuggingFaceEmbeddings
import torch


class Plugin(PluginCore):
    _embeddings: HuggingFaceEmbeddings

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)

        self.meta = Meta(
            name="HuggingFaceEmbeddings Plugin",
            description="Plugin for HuggingFace Embeddings",
            version="0.0.1",
        )

    def invoke(self, config_data: dict):
        device = "cpu"

        if str(config_data.get("device", "auto").lower()) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._embeddings = HuggingFaceEmbeddings(
            model_name=str(Path(config_data["model_path"])),
            model_kwargs={"device": device},
        )

        return self

    def getEmbeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings

    def encode(self, texts: list[str]) -> list:
        return self._embeddings.embed_documents(texts)
