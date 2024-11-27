from logging import Logger

from common.plugin.helper import LLMPluginCore
from common.plugin.model import Meta
from common.request.model import LLMRequestModel, OpenAIRequestModel

import requests
import aiohttp
from dateutil import parser
import json


class Plugin(LLMPluginCore):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name="Remote Ollama LLM Plugin",
            description="Remote Ollama LLM plugin template",
            version="0.0.1",
        )

    def invoke(self, config_data: dict):
        self.headers = {"Content-Type": "application/json"}

        self.model = {}
        self.llm_host = config_data["host"]
        self.llm_completion_url = f"{self.llm_host}/api/generate"
        self.llm_chat_completions_url = f"{self.llm_host}/v1/chat/completions"

        return self

    def models(self) -> str:
        llm_models_url = f"{self.llm_host}/api/tags"
        with requests.get(llm_models_url, headers=self.headers) as response:
            response.raise_for_status()

        raw_info = response.json()

        if len(raw_info["models"]) == 0:
            raise ValueError("No models exist!")

        models = []

        for info in raw_info["models"]:
            models.append(
                {
                    "id": info["name"],
                    "object": "model",
                    "created": int(parser.parse(info["modified_at"]).timestamp()),
                    "owned_by": "ollama",
                }
            )

        result = {"object": "list", "data": models}

        return json.dumps(result)

    def completion(self, request: LLMRequestModel) -> str:
        with requests.post(
            self.llm_completion_url, data=request.json(exclude_none=True), headers=self.headers
        ) as response:
            response.raise_for_status()

        return response.text

    async def stream_completion(self, request: LLMRequestModel):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.llm_completion_url,
                data=request.json(exclude_none=True),
                headers=self.headers,
                stream=True,
            ) as response:
                async for chunk in response.iter_content(1024):
                    if chunk:
                        yield chunk

    def chat_completions(self, request: OpenAIRequestModel) -> str:
        with requests.post(
            self.llm_chat_completions_url, data=request.json(exclude_none=True), headers=self.headers
        ) as response:
            response.raise_for_status()

        return response.text

    async def stream_chat_completions(self, request: OpenAIRequestModel):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.llm_chat_completions_url, data=request.json(exclude_none=True)
            ) as response:
                async for chunk in response.content.iter_any():
                    yield chunk
