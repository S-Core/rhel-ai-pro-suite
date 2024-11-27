import os
from common.plugin.helper import ChunkerPluginCore
from common.plugin.model import Meta
from common.exceptions import OSSRagException, ErrorCode
from typing import Any, List, Type
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from logging import Logger

class Plugin(ChunkerPluginCore):
    chunk_size : int
    _chunk_overlap : int
    _tiktoken_encoding_name : str
    _text_splitter : RecursiveCharacterTextSplitter

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self._logger.info("ChunkerPluginCore init...")
        self.meta = Meta(
            name="Chunker Plugin",
            description="Chunker plugin template",
            version="0.0.1",
        )

    def invoke(self, config_data: dict):
        self._logger.info("ChunkerPluginCore invoke...")
        self.chunk_size = config_data["chunk_size"]
        self._chunk_overlap = config_data["chunk_overlap"]
        self._tiktoken_encoding_name = config_data["tiktoken_encoding_name"]
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name = self._tiktoken_encoding_name,
                chunk_size = self.chunk_size,
                chunk_overlap = self._chunk_overlap,
        )
        return self
    async def chunk(self, **kwargs: Any) -> List[Document]:
        tid = kwargs.get("tid", None)
        document = kwargs.get("document", None)
        self._logger.debug("TID:%s Split content to chunks...", tid)
        try:
            documents = list()
            chunks = self._text_splitter.split_text(document.content)
            metadata = document.metadata
            for chunk in chunks:
                new_doc = Document(page_content=chunk,
                                metadata={
                                    "source": metadata.source,
                                    "name": metadata.name,
                                    "domain": metadata.domain,
                                    "author": metadata.author,
                                }
                            )
                documents.append(new_doc)
            self._logger.debug("TID:%s, 분할된 chunk 개수: %d", tid, len(documents))
            return documents
        except Exception as e:
            self._logger.error("TID:%s, %s", tid, e, exc_info=True)
            raise OSSRagException(ErrorCode.CHUNKER_UNKNOWN, e) from e
