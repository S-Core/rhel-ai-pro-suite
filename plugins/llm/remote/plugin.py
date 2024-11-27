from logging import Logger

from common.plugin.helper import LLMPluginCore
from common.plugin.model import Meta
from common.request.model import LLMRequestModel, OpenAIRequestModel

import requests
import aiohttp


class Plugin(LLMPluginCore):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name="Remote LLM Plugin",
            description="Remote LLM plugin template",
            version="0.0.1",
        )

    def invoke(self, config_data: dict) -> None:
        self.headers = {"Content-Type": "application/json"}

        api_key = config_data.get("api_key", None)

        if api_key is not None:
            self.headers["Authorization"] = "Bearer " + api_key

        self.llm_host = config_data["host"]
        self.llm_completion_url = f"{self.llm_host}/v1/completions"
        self.llm_chat_completions_url = f"{self.llm_host}/v1/chat/completions"

        return self

    def models(self) -> str:
        llm_models_url = f"{self.llm_host}/v1/models"
        with requests.get(llm_models_url, headers=self.headers) as response:
            response.raise_for_status()

        return response.text

    def completion(self, request: LLMRequestModel) -> str:
        with requests.post(
            self.llm_completion_url, json=request.dict(exclude_none=True), headers=self.headers
        ) as response:
            response.raise_for_status()

        return response.text

    def stream_completion(self, request: LLMRequestModel):
        with requests.post(
            self.llm_completion_url, json=request.dict(exclude_none=True), headers=self.headers, stream=True
        ) as response:
            for chunk in response.iter_content(1024):
                if chunk:
                    yield chunk

    def chat_completions(self, request: OpenAIRequestModel) -> str:
        with requests.post(
            self.llm_chat_completions_url, json=request.dict(exclude_none=True), headers=self.headers
        ) as response:
            response.raise_for_status()
                

        return response.text

    async def stream_chat_completions(self, request: OpenAIRequestModel):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.llm_chat_completions_url, json=request.dict(exclude_none=True), headers=self.headers
            ) as response:
                async for chunk in response.content.iter_any():
                    yield chunk
