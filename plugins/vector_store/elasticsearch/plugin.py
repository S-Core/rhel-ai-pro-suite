from logging import Logger
from pathlib import Path

from common.plugin.helper import VectorStorePluginCore
from common.plugin.model import Meta
from common.exceptions import OSSRagException, ErrorCode

from elasticsearch import Elasticsearch, helpers, NotFoundError
from elasticsearch_dsl import UpdateByQuery
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchRetriever, ElasticsearchStore
from langchain.retrievers.ensemble import EnsembleRetriever
from typing import Any, Dict, List, Callable, Tuple
from functools import partial

from elasticsearch.helpers import BulkIndexError


class Plugin(VectorStorePluginCore):
    vector_store: ElasticsearchStore
    _embeding_model: Any
    _client: Elasticsearch
    index_name: str
    _text_field: str
    _dense_vector_field: str

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name="Vector Store Elasticsearch Plugin",
            description="Vector Store Elasticsearch plugin template",
            version="0.0.1",
        )

    def invoke(self, config_data: dict, embeding_model: Any):
        client_kwargs = {"hosts": config_data["hosts"], "verify_certs": False}

        if "user" in config_data and "password" in config_data:
            client_kwargs["basic_auth"] = (config_data["user"], config_data["password"])

        if "ca_certs" in config_data:
            client_kwargs["ca_certs"] = str(Path(config_data["ca_certs"]))

        self._client = Elasticsearch(**client_kwargs)
        self._embeding_model = embeding_model
        self.index_name = config_data["index_name"]
        self._text_field = config_data.get("text_field", "text")
        self._dense_vector_field = config_data.get("dense_vector_field", "vector")
        self.vector_store = ElasticsearchStore(
            es_connection=self._client,
            index_name=self.index_name,
            embedding=embeding_model,
        )

        return self

    def _createUpdateByQuery(
        self, index_name: str, updateRequest: Any
    ) -> UpdateByQuery:
        ubq = UpdateByQuery(using=self._client, index=index_name)
        filter_dict = updateRequest.filter.dict()
        update_dict = updateRequest.update.dict()

        script_source = "; ".join(
            [f"ctx._source.metadata.{field} = params.{field}" for field in update_dict]
        )
        params = {field: value for field, value in update_dict.items()}
        ubq = ubq.script(source=script_source, params=params)
        for field, value in filter_dict.items():
            ubq = ubq.query("match", **{f"metadata.{field}": value})
        for field, value in update_dict.items():
            ubq = ubq.exclude("match", **{f"metadata.{field}": value})
        return ubq

    def updateByQuery(self, **kwargs: Any) -> Any:
        try:
            tid = kwargs.get("tid", None)
            self._logger.debug(f"TID:{tid}, Input:[{kwargs}]")
            index_name = kwargs.get("index_name", None)
            if not index_name:
                index_name = self.index_name
            updateRequest = kwargs.get("updateRequest", None)
            if updateRequest is None:
                self._logger.error("TID:%s updateRequest is none", tid)
                raise OSSRagException(ErrorCode.VECTOR_STORE_INVALID_INPUT_VALUE)
            ubq = self._createUpdateByQuery(index_name, updateRequest)
            response = ubq.execute()
            return response
        except Exception as e:
            self._logger.error("TID:%s, %s", tid, e)
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN)

    async def index_documents(self, **kwargs: Any) -> str:
        tid = kwargs.get("tid", None)
        index_name = kwargs.get("index_name", None)
        chunks = kwargs.get("chunks", 0)
        if not index_name:
            index_name = self.index_name
        try:
            await self.vector_store.afrom_documents(
                chunks,
                embedding=self._embeding_model,
                es_connection=self._client,
                index_name=index_name,
                query_field=self._text_field,
                vector_query_field=self._dense_vector_field,
                strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
            )
            return index_name
        except BulkIndexError as e:
            self._logger.error("TID:%s, %s[detail:%s]", tid, e, e.errors)
            raise OSSRagException(ErrorCode.VECTOR_STORE_INDEX_FAIL, e) from e
        except Exception as e:
            self._logger.error("TID:%s, %s", tid, e)
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

    def bulk(self, **kwargs: Any) -> str:
        tid = kwargs.get("tid")
        index_name = kwargs.get("index_name")
        documents = kwargs.get("documents")

        try:
            helpers.bulk(self._client, documents, index=index_name)
        except BulkIndexError as e:
            self._logger.error(f"TID: {tid}, {e}[detail:{e.errors}]")
            raise OSSRagException(ErrorCode.VECTOR_STORE_INDEX_FAIL, e) from e
        except Exception as e:
            self._logger.error(f"TID: {tid}, {e}")
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

    async def async_bulk(self, **kwargs: Any) -> str:
        tid = kwargs.get("tid")
        index_name = kwargs.get("index_name")
        documents = kwargs.get("documents")

        try:
            await helpers.async_bulk(self._client, documents, index=index_name)
        except BulkIndexError as e:
            self._logger.error(f"TID: {tid}, {e}[detail:{e.errors}]")
            raise OSSRagException(ErrorCode.VECTOR_STORE_INDEX_FAIL, e) from e
        except Exception as e:
            self._logger.error(f"TID: {tid}, {e}")
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

    def search(self, **kwargs: Any) -> List:
        tid = kwargs.get("tid")
        index_name = kwargs.get("index_name")
        body = kwargs.get("body")

        try:
            response = self._client.search(index=index_name, body=body)

            hits = response["hits"]["hits"]
            documents = [hit["_source"] for hit in hits]
        except Exception as e:
            self._logger.error(f"TID: {tid}, {e}")
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

        return documents

    def similarity_search(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k, **kwargs)

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return await self.vector_store.asimilarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )

    def _create_query_dsl(
        self,
        search_query: str,
        embedding: Any = None,
        k: int = 5,
        num_candidates: int = 100,
        text_field: str = "text",
        dense_vector_field: str = "vector",
        use_vector: bool = True,
        use_bm25: bool = False,
        size: int = 10,
    ) -> Dict:
        if embedding is None:
            embedding = self._embeding_model

        vector = embedding.embed_query(search_query)
        query_dsl = {}

        if use_vector:
            query_dsl["knn"] = {
                "field": dense_vector_field,
                "query_vector": vector,
                "k": k,
                "num_candidates": num_candidates,
            }

        if use_bm25:
            query_dsl["query"] = {
                "match": {
                    text_field: {
                        "query": search_query,
                    }
                }
            }

        query_dsl["size"] = size if size > 0 else 0

        return query_dsl

    def _setup_retriever(
        self,
        client: Any,
        index_name: str,
        text_field: str,
        create_query_dsl: Callable[..., dict],
    ) -> ElasticsearchRetriever:
        body_func = lambda q: create_query_dsl(q)
        retriever = ElasticsearchRetriever(
            es_client=client,
            body_func=body_func,
            index_name=index_name,
            content_field=text_field,
        )
        return retriever

    def create_lexical_retriver(
        self,
        size: int = 10,
    ) -> ElasticsearchRetriever:
        query_dsl_func = partial(
            self._create_query_dsl,
            use_bm25=True,
            text_field=self._text_field,
            size=size,
        )
        retriever = self._setup_retriever(
            self._client, self.index_name, self._text_field, query_dsl_func
        )
        return retriever

    def lexical_search(
        self,
        query: str,
        size: int = 10,
    ) -> List[Document]:
        try:
            return self.create_lexical_retriver(size=size).invoke(query)
        except NotFoundError as e:
            self._logger.error(e)
            return []
        except Exception as e:
            self._logger.error(e)
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

    async def alexical_search(
        self,
        query: str,
        size: int = 10,
    ) -> List[Document]:
        try:
            return await self.create_lexical_retriver(size=size).ainvoke(query)
        except NotFoundError as e:
            self._logger.error(e)
            return []
        except Exception as e:
            self._logger.error(e)
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

    def create_semantic_retriver(
        self,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
    ) -> ElasticsearchRetriever:
        query_dsl_func = partial(
            self._create_query_dsl,
            use_vector=True,
            dense_vector_field=self._dense_vector_field,
            k=k,
            num_candidates=num_candidates,
            size=size,
        )
        retriever = self._setup_retriever(
            self._client, self.index_name, self._text_field, query_dsl_func
        )
        return retriever

    def semantic_search(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
    ) -> List[Document]:
        try:
            return self.create_semantic_retriver(
                k=k, num_candidates=num_candidates, size=size
            ).invoke(query)
        except NotFoundError as e:
            self._logger.error(e)
            return []
        except Exception as e:
            self._logger.error(e)
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

    async def asemantic_search(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
    ) -> List[Document]:
        try:
            return await self.create_semantic_retriver(
                k=k, num_candidates=num_candidates, size=size
            ).ainvoke(query)
        except NotFoundError as e:
            self._logger.error(e)
            return []
        except Exception as e:
            self._logger.error(e)
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

    def create_hybrid_retriver(
        self,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
        weights: List[float] = [0.50, 0.50],
    ) -> EnsembleRetriever:
        bm25_retriver = self.create_lexical_retriver(size=size)
        vector_retriver = self.create_semantic_retriver(
            k=k, num_candidates=num_candidates, size=size
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriver, vector_retriver],
            weights=weights,
        )
        return ensemble_retriever

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
        weights: List[float] = [0.50, 0.50],
    ) -> List[Document]:
        try:
            docs = self.create_hybrid_retriver(
                k=k, num_candidates=num_candidates, size=size, weights=weights
            ).invoke(query)
        except NotFoundError as e:
            self._logger.error(e)
            return []
        except Exception as e:
            self._logger.error(e)
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

        return docs[0:size]

    async def ahybrid_search(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        size: int = 10,
        weights: List[float] = [0.50, 0.50],
    ) -> List[Document]:
        try:
            docs = await self.create_hybrid_retriver(
                k=k, num_candidates=num_candidates, size=size, weights=weights
            ).ainvoke(query)
        except NotFoundError as e:
            self._logger.error(e)
            return []
        except Exception as e:
            self._logger.error(e)
            raise OSSRagException(ErrorCode.VECTOR_STORE_UNKNOWN, e) from e

        return docs[0:size]
