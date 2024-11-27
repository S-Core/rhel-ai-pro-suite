from logging import Logger
from pathlib import Path

from common.plugin.helper import PluginCore
from common.plugin.model import Meta
from common.request.model import RerankRequestModel

from typing import List, Any
from fastapi import APIRouter

import json

from FlagEmbedding import FlagReranker


class Plugin(PluginCore):
    _model: FlagReranker

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)

        self.meta = Meta(
            name="bge-reranker-reranker Plugin",
            description="bge-reranker plugin based transformers",
            version="0.0.1",
        )

    def invoke(self, config_data: dict):
        model_path = str(Path(config_data["model_path"]))

        self._model = FlagReranker(model_path, use_fp16=True)

        return self

    def registerAPIs(self, node: Any):
        router = APIRouter()
        router.add_api_route("/v1/rerank", self._rerank, methods=["POST"])
        return router

    def _rerank(self, request: RerankRequestModel):
        result = self.rerank(request.question, request.contexts, request.k)
        return json.dumps(result, ensure_ascii=False)

    def rerank(self, query: str, contexts: List[str], k: int = 3) -> List[dict]:
        pairs = [[query, context] for context in contexts]

        scores = self._model.compute_score(pairs, normalize=True)

        result = [
            {"context": a, "score": b, "index": i}
            for i, (a, b) in enumerate(zip(contexts, scores))
        ]

        sorted_result = sorted(result, key=lambda x: float(x["score"]), reverse=True)

        self._logger.debug("-" * 32)
        self._logger.debug("* reranked_contexts:")
        self._logger.debug(sorted_result)
        self._logger.debug("-" * 32)

        return sorted_result[:k]
