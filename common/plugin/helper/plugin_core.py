from abc import abstractmethod
from logging import Logger
from typing import List, Dict, Tuple, Optional, Any

from common.plugin.model import Meta
from common.request.model import LLMRequestModel, OpenAIRequestModel

from langchain_core.documents import Document
from fastapi import APIRouter


class IPluginRegistry(type):
    plugin_registries: List[type] = list()

    def __init__(cls, name, bases, attrs):
        super().__init__(cls)
        if name != "PluginCore":
            IPluginRegistry.plugin_registries.append(cls)


class PluginCore(object, metaclass=IPluginRegistry):

    meta: Optional[Meta]
    router: Optional[APIRouter] = None

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    @abstractmethod
    def invoke(self, config_data: dict, **kwargs) -> Any:
        pass

    @abstractmethod
    def registerAPIs(self, node: Any) -> APIRouter:
        pass


class VectorStorePluginCore(PluginCore):

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[Document]:
        pass

    @abstractmethod
    def updateByQuery(self, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    async def index_documents(self, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def bulk(self, **kwargs: Any) -> str:
        pass

    @abstractmethod
    async def async_bulk(self, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def search(self, **kwargs: Any) -> List:
        pass

    @abstractmethod
    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    def lexical_search(
        self,
        query: str,
        size: int = 10,
    ) -> List[Document]:
        pass

    @abstractmethod
    async def alexical_search(
        self,
        query: str,
        size: int = 10,
    ) -> List[Document]:
        pass

    @abstractmethod
    def semantic_search(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
    ) -> List[Document]:
        pass

    @abstractmethod
    async def asemantic_search(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
    ) -> List[Document]:
        pass

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
        weights: List[float] = [0.50, 0.50],
    ) -> List[Document]:
        pass

    @abstractmethod
    async def ahybrid_search(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
        weights: List[float] = [0.50, 0.50],
    ) -> List[Document]:
        pass


class LLMPluginCore(PluginCore):

    @abstractmethod
    def models(self) -> str:
        pass

    @abstractmethod
    def completion(self, request: LLMRequestModel) -> str:
        pass

    @abstractmethod
    async def stream_completion(self, request: LLMRequestModel):
        pass

    @abstractmethod
    def chat_completions(self, request: OpenAIRequestModel) -> str:
        pass

    @abstractmethod
    async def stream_chat_completions(self, request: OpenAIRequestModel):
        pass

class ChunkerPluginCore(PluginCore):
    @abstractmethod
    async def chunk(self, **kwargs: Any) -> List[Document]:
        """텍스트를 청킹하여 리스트로 반환하는 추상 메서드."""
        pass

class FilterPluginCore(PluginCore):
    @abstractmethod
    def apply_filter(self, **kwargs: Any) -> List[Any]:
        pass
