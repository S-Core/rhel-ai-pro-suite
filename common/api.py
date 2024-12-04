from common.exceptions import  OSSRagException, ErrorCode
from fastapi import APIRouter, Request, UploadFile, Depends, File, Query, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from .plugin.helper import (
    ChunkerPluginCore,
    LLMPluginCore,
    PluginCore,
    VectorStorePluginCore,
    FilterPluginCore
)
from logging import Logger
from typing import List, Optional, Tuple
from .request.model import (
    DocumentModel,
    DocumentMetaDataModel,
    LLMRequestModel,
    Message,
    OpenAIRequestModel,
    parse_document_meta_data_model,
    UpdateDocumentRequestModel
)
import asyncio
import json
import math
import validators


MEDIA_TYPE_EVENT_STREAM = "text/event-stream"
TEXT_PREFIX_SOURCE = " - Sources : "
TEXT_SEARCH_FAIL = "Unfortunately, we couldn't find any documents related to your query. Please try rephrasing or asking a different question."
PROMPT_TEMPLATE_QA = r"""
    Answer the questions in Korean based solely on the following context.:
    ----
    {context}

    ----
    Question: {question}
"""
PROMPT_TEMPLATE_USER = r"""
    ----
    Context: {context}

    ----
    Question: {question}
"""
PROMPT_TEMPLATE_CONTEXT_STRICT = r"""
You are a knowledgeable AI assistant that answers strictly based on the provided context.

Rules:
1. Use only the information in the context.
2. Do not reference external knowledge or make assumptions.
3. If the context lacks the necessary information, respond:
   "The provided context does not contain the requested information."
5. Avoid fabricating answers; respond only based on available information.
6. Exclude redundant content in your responses to ensure clarity and conciseness.

Context: {context}

Question: {question}
"""
PROMPT_TEMPLATE_CONTEXT_FIRST = r"""
You are a knowledgeable AI assistant that combines provided context with your general knowledge to give comprehensive answers.

Rules:
1. If relevant context is provided, prioritize and clearly reference this information
2. You may combine context information with general knowledge for more complete answers
3. If the context contains information that conflicts with your knowledge, prioritize the context
4. Always indicate the source of information (context vs. general knowledge)
5. Avoid fabricating answers; respond only based on available information.
6. Exclude redundant content in your responses to ensure clarity and conciseness.

Context: {context}

Question: {question}
"""
class APIManager:
    _logger: Logger
    _llm: LLMPluginCore
    _vector_store: VectorStorePluginCore
    _reranker: PluginCore
    _chunker: ChunkerPluginCore
    _filter: FilterPluginCore
    _options: dict

    def __init__(self, **kwargs):
        self._logger = kwargs["logger"]
        self._llm = kwargs["llm"]
        self._vector_store = kwargs["vector_store"]
        self._reranker = kwargs["reranker"] if "reranker" in kwargs else None
        self._chunker = kwargs["chunker"]
        self._filter = kwargs["filter"]
        options = kwargs["options"]
        self.options = {
            "strict_answer": (
                options["strict_answer"]
                if "strict_answer" in options and options["strict_answer"] is not None
                else False
            ),
            "source_visible": (
                str(options["source_visible"]).lower() == "true"
                if "source_visible" in options and options["source_visible"] is not None
                else True
            ),
            "context_visible": (
                str(options["context_visible"]).lower() == "true"
                if "context_visible" in options
                and options["context_visible"] is not None
                else False
            ),
            "search_size": (
                int(options["search_size"])
                if "search_size" in options and options["search_size"] is not None
                else 10
            ),
            "context_size": (
                int(options["context_size"])
                if "context_size" in options and options["context_size"] is not None
                else 3
            ),
            "min_similarity": (
                float(options["min_similarity"])
                if "min_similarity" in options
                and math.isnan(float(options["min_similarity"])) is False
                else 0.0
            ),
            "retrieval_type": (
                str(options["retrieval_type"])
                if "retrieval_type" in options and options["retrieval_type"] is not None
                else "semantic"
            ),
        }

        self.router = APIRouter()
        self.router.add_api_route("/v1/models", self._models, methods=["GET"])
        self.router.add_api_route("/v1/completions", self._completion, methods=["POST"])
        self.router.add_api_route(
            "/v1/chat/completions", self._chat_completions, methods=["POST"]
        )
        self.router.add_api_route(
            "/v1/documents/texts", self._upload_document_text, methods=["POST"]
        )
        self.router.add_api_route(
            "/v1/documents/files", self._upload_document_file, methods=["POST"]
        )
        self.router.add_api_route(
            "/v1/documents", self._update_document, methods=["PATCH"]
        )

    def _models(self):
        return Response(content=self._llm.models())

    async def _completion(self, request: LLMRequestModel):

        search_result = getattr(
            self._vector_store,
            (
                "hybrid_search"
                if self.options["retrieval_type"] == "hybrid"
                else (
                    "lexical_search"
                    if self.options["retrieval_type"] == "lexical"
                    else "semantic_search"
                )
            ),
        )(request.prompt, size=self.options["search_size"])

        content = ""
        for result in search_result:
            content += result.page_content + "\n"

        prompt = PromptTemplate(
            input_variables=["context", "question"], template=PROMPT_TEMPLATE_QA
        )

        print(prompt.format(context=content, question=request.prompt))

        request.prompt = prompt.format(context=content, question=request.prompt)

        if request.stream is not True:
            return Response(content=self._llm.completion(request))
        else:
            return StreamingResponse(
                content=self._llm.stream_completion(request),
                media_type=MEDIA_TYPE_EVENT_STREAM,
            )
    async def _update_document(self, request: Request,
        updateRequest: UpdateDocumentRequestModel = Body(..., description="Represents the request to update a document."),
        index_name: Optional[str] = Query(None, description="Name of the index where the document resides.")
    ):
        """
        Update a document in the system.

        This endpoint processes the request to update a document's status or other attributes.
        The update request must include the filter criteria and the fields to be updated.

        Parameters:
            updateRequest (UpdateDocumentRequestModel):
                A Pydantic model representing the request to update a document,
                which includes a filter to identify the document and the update data (e.g., status).

            index_name (Optional[str], default None):
                The name of the index where the document resides. This is an optional query parameter.

        Returns:
            dict: A dictionary containing the updated document information.

        Description:
            - **updateRequest**: A request body containing the filter criteria and update fields for the document.
            - **index_name**: An optional query parameter that specifies the index name where the document is stored.
        """
        try:
            response = self._vector_store.updateByQuery(tid=request.state.transaction_id,
                updateRequest=updateRequest,
                index_name=index_name)

            return JSONResponse(
                status_code = 200,
                content = {
                    "messsage" : "success",
                    "updated_count": response.updated
                }
            )
        except (OSSRagException) as e:
            raise e
        except Exception as e:
            raise OSSRagException(ErrorCode.UPDATE_DOCUMENT_MODEL_NOT_FOUND, e) from e

    async def _upload_document_text(self, request: Request, document: DocumentModel):
        try:
            self._logger.debug("TID: %s, request: %s", request.state.transaction_id, document)
            doc = await self._chunker.chunk(tid=request.state.transaction_id, document=document)
            index_name = await self._vector_store.index_documents(chunks=doc, index_name=document.index_name, tid=request.state.transaction_id)
            return JSONResponse(
                status_code = 200,
                content = {
                    "messsage" : "success",
                    "index_name" : index_name
                }
            )
        except (OSSRagException) as e:
            raise e
        except Exception as e:
            raise OSSRagException(ErrorCode.UPLOAD_DOCUMENT_UNKNOWN, e) from e
            
    async def _upload_document_file(
            self, request: Request, index_name: Optional[str] = None,
            documentMeta: DocumentMetaDataModel = Depends(parse_document_meta_data_model),
            file: UploadFile = File(...)):
        try:
            self._logger.debug("TID: %s, request: %s", request.state.transaction_id, documentMeta)
            content = await file.read()
            document = DocumentModel(metadata=documentMeta, content=content, index_name=index_name)
            doc = await self._chunker.chunk(tid=request.state.transaction_id, document=document)
            index_name = await self._vector_store.index_documents(chunks=doc, index_name=document.index_name, tid=request.state.transaction_id)
            return JSONResponse(
                status_code = 200,
                content = {
                    "messsage" : "success",
                    "index_name" : index_name
                }
            )
        except OSSRagException as e:
            raise e
        except Exception as e:
            raise OSSRagException(ErrorCode.UPLOAD_DOCUMENT_UNKNOWN, e) from e

    async def _chat_completions(self, http_request: Request, request: OpenAIRequestModel):

        ordered_docs = []

        for message in request.messages:
            if message.role == "user":
                query = message.content
            elif message.role == "assistant":
                prefix_source_index = message.content.rfind(TEXT_PREFIX_SOURCE)
                if prefix_source_index != -1:
                    message.content = message.content[:prefix_source_index]

        retrieved_docs_with_score = await getattr(
            self._vector_store,
            (
                "ahybrid_search"
                if self.options["retrieval_type"] == "hybrid"
                else (
                    "alexical_search"
                    if self.options["retrieval_type"] == "lexical"
                    else "asemantic_search"
                )
            ),
        )(query, size=self.options["search_size"])

        if self._reranker is not None and len(retrieved_docs_with_score) > 0:
            reranked_docs = self._reranker.rerank(
                query,
                [doc.page_content for doc in retrieved_docs_with_score],
                self.options["context_size"],
            )

            ordered_docs = [
                {
                    "context": retrieved_docs_with_score[doc["index"]],
                    "score": doc["score"],
                }
                for doc in reranked_docs
            ]
        else:
            self._logger.debug(
                "-" * 30
                + "[APIManager]"
                + f" similarity_search_with_relevance_scores (query : {query}, k : {self.options['search_size']})"
                + "-" * 30
            )

            ordered_docs = [
                {"context": doc, "score": doc.metadata["_score"]}
                for doc in retrieved_docs_with_score
            ]
            ordered_docs = ordered_docs[: self.options["context_size"]]

        search_content = ""
        docs_with_score = []
        filtered = False
        filterd_docs = self._filter.apply_filter(documents=ordered_docs, tid=http_request.state.transaction_id)
        if len(filterd_docs) != len(ordered_docs):
            filtered = True
            self._logger.debug("TID: %s, document filtered[searched count[%d], filtered count[%d]]",
                http_request.state.transaction_id, len(ordered_docs), len(filterd_docs))
        for doc in filterd_docs:
            if float(doc["score"]) >= self.options["min_similarity"]:
                search_content += doc["context"].page_content + "\n----\n"
                docs_with_score.append((doc["context"], doc["score"]))
        if self.options["strict_answer"]:
            if not search_content and filtered is False:

                if not request.stream:
                    return Response(
                        content=f'{{"choices":[{{"finish_reason":"stop","index":0,"message":{{"content":"{TEXT_SEARCH_FAIL}","role":"assistant"}}}}]}}'
                    )
                else:
                    return StreamingResponse(
                        content=self.__no_search_result_stream_generate(
                            TEXT_SEARCH_FAIL
                        ),
                        media_type=MEDIA_TYPE_EVENT_STREAM,
                    )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE_CONTEXT_FIRST if filtered else PROMPT_TEMPLATE_CONTEXT_STRICT
        )

        query_prompt = prompt.format(context=search_content, question=query)

        request.messages.append(Message(role="user", content=query_prompt))

        self._logger.debug(request)

        if not request.stream:
            return Response(content=self._llm.chat_completions(request))
        else:
            return StreamingResponse(
                content=self.__stream_chat_completions(http_request, request, docs_with_score),
                media_type=MEDIA_TYPE_EVENT_STREAM,
            )

    async def __no_search_result_stream_generate(self, no_search_result_response: str):
        for chunk in no_search_result_response:
            yield b'data: {"choices":[{"delta":{"content":"' + chunk.encode(
                "utf-8"
            ) + b'"},"finish_reason":null,"index":0}]}\n\n'
            await asyncio.sleep(0.01)
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n'

    def _get_rag_source(self, docs_with_score: List[Tuple[Document, float]]) -> bytes:
        source_link_set = set()
        for doc, score in docs_with_score:
            domain = doc.metadata.get("_source", {}).get("metadata", {}).get("domain")
            if domain:
                source_link_set.add(domain)

        source_link = ""
        for i, source_link_str in enumerate(source_link_set):
            source_link += source_link_str
            if i < len(source_link_set) - 1:
                source_link += ", "

        data = {
            "choices": [
                {
                    "delta": {
                        "content": source_link,
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "object":"rag.source"
        }
        return b"data: " + json.dumps(
            data, ensure_ascii=False
        ).encode("utf-8") + b"\n\n"

    def _get_rag_context(self, docs_with_score: List[Tuple[Document, float]]) -> bytes:
        source_content = ""
        for i, doc_with_score in enumerate(docs_with_score):
            doc, score = doc_with_score
            source_content += (
                doc.page_content
                + " (**Score**:"
                + str(score)
                + ")"
            )
            if i < len(docs_with_score) - 1:
                source_content += "\n\n "

        data = {
            "choices": [
                {
                    "delta": {
                        "content": source_content,
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "object":"rag.context"
        }

        return b"data: " + json.dumps(
            data, ensure_ascii=False
        ).encode("utf-8") + b"\n\n"

    async def __stream_chat_completions(
        self, http_request: Request, request: OpenAIRequestModel, docs_with_score: List[Tuple[Document, float]]
    ):
        tid = http_request.state.transaction_id
        add_context_source = False
        async for chunk in self._llm.stream_chat_completions(request):
            # Hooking chunk data to record the source
            if add_context_source is False and \
                (self.options["context_visible"] or self.options["source_visible"]) and \
                len(docs_with_score) > 0: #in case already trained all data. No data to extract source and context

                self._logger.debug("TID: %s, chunk[ %s ]", tid, chunk)
                chunk_datas = chunk.split(b"data:")
                for chunk_data in chunk_datas:
                    if not chunk_data:
                        continue
                    try:
                        chunk_json = json.loads(chunk_data)

                        if not (chunk_json.get("choices") and
                            chunk_json["choices"][-1].get("finish_reason") in {"stop", "length"}
                        ):
                            continue
                        if self.options["source_visible"]:
                            yield self._get_rag_source(docs_with_score)

                        if self.options["context_visible"]:
                            yield self._get_rag_context(docs_with_score)

                        add_context_source = True # responsed all sources and contexts above.
                        break
                    except json.JSONDecodeError as e:
                        self._logger.debug("TID : %s, Invalid Json Format(%s), exception(%s)", tid, chunk_data, e)
                        continue
                    except Exception as e:
                        self._logger.error("TID : %s, Exception occurred: %s", tid, e, exc_info=True)
                        continue

            yield chunk
