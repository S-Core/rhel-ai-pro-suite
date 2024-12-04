import os
import json

from logging import Logger
from typing import Any
from common.plugin.helper import PluginCore
from common.exceptions import OSSRagException, ErrorCode

from util import AppConfig
from .evaluator import Evaluator
from .model import (
    TestSetRequestModel,
    TestSetResponseModel,
    EvaluationRequestModel,
    EvaluationResponseModel,
)
from node import Node

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse


# Unless you assign "RAGAS_DO_NOT_TRACK" flag to True,
# RAGAS tracks information that can be used to identify a user or company.
os.environ["RAGAS_DO_NOT_TRACK"] = "TRUE"


class Plugin(PluginCore):
    _logger: Logger
    evaluator: Evaluator = None

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self._app_config = AppConfig().get()

    def invoke(self, config_data: dict):
        self._config = config_data
        return self

    def registerAPIs(self, node: Node):
        router = APIRouter()

        if self.evaluator is None:
            default_llm = dict(self._app_config["plugins"]["llm"][self._config["llm"]])
            default_llm_model = default_llm.get("model", "")
            default_llm_host = default_llm["host"]
            default_llm_headers = default_llm.get("headers", node.get_llm().headers)

            critic_llm = {}
            critic_llm_config = self._config.get("critic_llm")
            if critic_llm_config is None:
                critic_llm["headers"] = default_llm_headers
            else:
                critic_llm["headers"] = dict(critic_llm_config).get("headers", {})

            self.evaluator = Evaluator(
                self._logger,
                chunk_size=node.get_chunker().chunk_size,
                embedding=node.get_embedding_model(),
                vector_store=node.get_vector_store(),
                llm=node.get_llm(),
                generator_llm={
                    "name": default_llm_model,
                    "url": default_llm_host,
                    "headers": default_llm_headers,
                },
                critic_llm={
                    **critic_llm,
                    "name": dict(self._config.get("critic_llm", {})).get("model", default_llm_model),
                    "url": dict(self._config.get("critic_llm", {})).get("host", default_llm_host),
                },
                metric_names=self._config["metrics"],
                retrieval_type=node._api_manager.options["retrieval_type"],
                document_index_name=node.get_vector_store().index_name,
                testset_index_name=self._config["testset_index_name"],
                evaluation_index_name=self._config["evaluation_index_name"],
            )

        router.add_api_route("/v1/qna/generate", self._generate_testset, methods=["POST"])
        router.add_api_route("/v1/qna/evaluate", self._evaluation, methods=["POST"])

        return router

    async def _generate_testset(self, request: Request, testset_request_model: TestSetRequestModel) -> StreamingResponse:
        self._logger.debug(f"TID: {request.state.transaction_id}, request: {testset_request_model}")

        try:
            return self.evaluator.generate_testset(tid=request.state.transaction_id, request=testset_request_model)
        except OSSRagException as e:
            raise e
        except Exception as e:
            raise OSSRagException(ErrorCode.EVALUATION_TESTSET_GENERATION_UNKNOWN, e) from e

    async def _evaluation(
        self, request: Request, evaluation_request_model: EvaluationRequestModel
    ) -> EvaluationResponseModel:
        self._logger.debug(f"TID: {request.state.transaction_id}, request: {evaluation_request_model}")

        try:
            return self.evaluator.evaluate(tid=request.state.transaction_id, request=evaluation_request_model)
        except OSSRagException as e:
            raise e
        except Exception as e:
            raise OSSRagException(ErrorCode.EVALUATION_UNKNOWN, e) from e
