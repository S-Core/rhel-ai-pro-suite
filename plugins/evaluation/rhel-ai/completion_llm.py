import json
import requests
from typing import Any, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class CompletionLLM(LLM):
    model_name: str = ""
    llm_url: str = ""
    system_prompt: str = ""
    headers: dict = {}

    def __init__(self, completion_llm: dict, system_prompt: str) -> None:
        super().__init__()
        self.model_name = completion_llm["name"]
        self.llm_url = completion_llm["url"] + "/v1/chat/completions"
        self.headers = completion_llm["headers"]
        self.system_prompt = system_prompt

    @property
    def _llm_type(self) -> str:
        return "CriticLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        with requests.post(
            self.llm_url,
            json.dumps(
                {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
            ),
            headers=self.headers,
        ) as r:
            r.raise_for_status()
            return r.json()["choices"][0]["message"][
                "content"
            ]  # get the response from the API

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": self.llm_url}
