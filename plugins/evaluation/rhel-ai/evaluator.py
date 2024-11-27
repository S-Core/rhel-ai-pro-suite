import re
import json
import yaml
from io import StringIO
import numpy as np

from logging import Logger
from typing import List, Dict, Any
from datetime import datetime
from datasets import Dataset
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from fastapi.responses import StreamingResponse

from common.plugin.helper import VectorStorePluginCore, LLMPluginCore
from common.request.model import LLMRequestModel, OpenAIRequestModel
from common.exceptions import OSSRagException, ErrorCode

from .model import (
    RequestModel,
    TestSetRequestModel,
    TestSetResponseModel,
    TestSetResultModel,
    TestSetMessageModel,
)

from .model import (
    EvaluationRequestModel,
    EvaluationResponseModel,
    EvaluationResultModel,
    EvaluationMessageModel,
)

from .model import (
    QnaModel,
    SeedExamplesModel,
    DocumentModel,
    QnaYamlModel,
)

from .completion_llm import CompletionLLM
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    noise_sensitivity_relevant,
    answer_similarity
)
from ragas.run_config import RunConfig


METRIC_MAPPING = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
    "context_entity_recall": context_entity_recall,
    "noise_sensitivity_relevant": noise_sensitivity_relevant,
    "answer_similarity": answer_similarity,
}

DEFAULT_SYSTEM_PROMPT = """You are an assistant for Q&A.
Please keep your answer concise.
"""

EXTRA_RAG_SYSTEM_PROMPT = """Please answer the questions by referring only to the following context.
Please use a maximum of 2 sentences for your answer.
If you don’t know, say you don’t know.
"""

SUMMARY_SYSTEM_PROMPT = """Summarize the content below in 100 characters or less.
Content: {content}"""


class Evaluator:
    
    def __init__(
        self,
        logger: Logger,
        embedding: Embeddings,
        vector_store: VectorStorePluginCore,
        llm: LLMPluginCore,
        critic_llm: Dict[str, Any],
        metric_names: List[str],
        retrieval_type: str,
        document_index_name: str,
        testset_index_name: str,
        evaluation_index_name: str,
        chunk_size: int = 512,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._logger = logger
        self.embedding = embedding
        self.vector_store = vector_store
        self.llm = llm
        self.critic_llm = critic_llm
        self.metric_names = metric_names
        self.metric_modules = [
            METRIC_MAPPING[name] for name in metric_names if name in METRIC_MAPPING
        ]
        self.retrieval_type = retrieval_type
        self.chunk_size = chunk_size
        self._system_prompt = system_prompt

        self.document_index_name = document_index_name
        self.testset_index_name = testset_index_name
        self.evaluation_index_name = evaluation_index_name

    def generate_testset(self, tid: str, request: TestSetRequestModel) -> StreamingResponse:
        target_model = self._get_target_model(tid, request)
        target_documents = self._get_target_testset_documents(tid, request)
        testsets = self._generate_testsets_from_documents(tid, request, target_model, target_documents)
        if request.qna_yaml.document_outline is None:
            document_outline = self._summarize_content(tid, target_model, target_documents)
        else:
            document_outline = request.qna_yaml.document_outline
        _, testset_documents, qna_yaml_model = self._get_processed_results(tid, request, testsets, target_model, document_outline)

        testset_index_name = (
            request.testset_index_name
            if request.testset_index_name is not None
            else self.testset_index_name
        )

        self.vector_store.bulk(tid=tid, index_name=testset_index_name, documents=testset_documents)

        yaml_data = yaml.dump(qna_yaml_model.dict(), allow_unicode=True, default_flow_style=False, sort_keys=False, width=100)
        
        # Contains YAML data in a StringIO object and returns it for streaming
        buffer = StringIO(yaml_data)
        
        return StreamingResponse(buffer, media_type="application/json")

    def _make_content_search_query(self, domain: str) -> Dict:
        query_dsl = {}

        query_dsl["query"] = {
            "bool": {
                "must": [
                    {
                        "match": {
                            "metadata.domain": domain
                        }
                    }
                ],
                "must_not": [
                    {
                        "match": {
                            "metadata.status": "trained"
                        }
                    }
                ]
            }
        }

        query_dsl["size"] = 10000

        return query_dsl
    
    def _get_target_testset_documents(self, tid: str, request: TestSetRequestModel) -> List:
        document_index_name = (
            request.document_index_name
            if request.document_index_name is not None
            else self.document_index_name
        )
        
        query_dsl = self._make_content_search_query(request.domain)

        target_documents = self.vector_store.search(tid=tid, index_name=document_index_name, body=query_dsl)

        if len(target_documents) < 5:
            self._logger.error(f"TID: {tid}, The number of documents is less than 5!")
            raise OSSRagException(ErrorCode.EVALUATION_DOCUMENTS_SIZE_EXCEPTION)

        return target_documents

    def _generate_testsets_from_documents(self, tid: str, request: TestSetRequestModel, target_model: str, target_documents: List):
        documents = [
            Document(
                page_content=document["text"],
                metadata=document["metadata"] if document["metadata"] is not None else {},
            )
            for document in target_documents
        ]

        system_prompt, _ = self._get_prompts(self._system_prompt)
        generator = TestsetGenerator.from_langchain(
            generator_llm=CompletionLLM(
                {
                    "name": target_model,
                    "url": self.critic_llm["url"],
                    "headers": self.critic_llm["headers"],
                },
                system_prompt,
            ),
            critic_llm=CompletionLLM(self.critic_llm, system_prompt),
            embeddings=self.embedding,
            chunk_size=self.chunk_size,
        )

        testsets = []
        for document in documents:
            testset = generator.generate_with_langchain_docs(
                [document],
                test_size=request.testset_size,
                distributions={
                    simple: 1.0,
                },
                raise_exceptions=False,
            )
            
            testsets.append(testset)

        if len(testsets) < 5:
            self._logger.error(f"TID: {tid}, The number of testset is less than 5!")
            raise OSSRagException(ErrorCode.EVALUATION_TESTSET_SIZE_EXCEPTION)

        return testsets
    
    def _extract_sentences(self, content: str) -> str:
        self._logger.info(f"The first and second sentences are extracted and used.")
        sentences = re.split(r'(?<=\.)\s+', content)
        sentence_0 = sentences[0].strip()
        if (len(sentences) >= 2):
            sentence = f"{sentence_0} {sentences[1].strip()}"
        self._logger.debug(f"Extracted Sentence: {sentence}")
        return sentence

    def _summarize_content(self, tid: str, target_model: str, target_documents: List) -> str:
        target_content = target_documents[0]["text"]

        system_prompt, _ = self._get_prompts(type="SUMMARY")

        prompt_template = PromptTemplate(
            input_variables=["content"], template=system_prompt
        )
        
        prompt_str = prompt_template.format(content=target_content)

        llm_request = LLMRequestModel(prompt=prompt_str, model=target_model)

        try:
            response = json.loads(self.llm.completion(llm_request))
        except Exception as e:
            self._logger.error(f"TID: {tid}, {e}", exc_info=True)
            self._logger.info("Content summary failed!!")
            return self._extract_sentences(target_content)

        # ollama
        if "response" in response:
            response_str = response["response"]
        # llama.cpp & vllm
        elif "choices" in response:
            response_str = response["choices"][0]["text"]
        else:
            self._logger.info(f"LLM responses cannot be analyzed, so summary sentences are extracted from the content.")
            self._logger.info(f"LLM Response: {response}")
            response_str = self._extract_sentences(target_content)

        return response_str
    
    def _get_qna_yaml_model(self, request: TestSetRequestModel, document_outline: str) -> QnaYamlModel:
        version = request.qna_yaml.version
        created_by = request.qna_yaml.created_by
        repo = request.qna_yaml.repo
        commit = request.qna_yaml.commit
        patterns = request.qna_yaml.patterns

        qna_yaml_model = QnaYamlModel(version=version,
                                      created_by=created_by,
                                      domain=request.domain,
                                      seed_examples=[],
                                      document_outline=document_outline,
                                      document=DocumentModel(commit=commit,
                                                             repo=repo,
                                                             patterns=patterns))
        
        return qna_yaml_model
    
    def _get_processed_results(self, tid: str, request: TestSetRequestModel, testsets: List, target_model: str, document_outline: str):
        qna_yaml_model = self._get_qna_yaml_model(request, document_outline)

        testset_results = []
        testset_documents = []
        try:
            for testset in testsets:
                contexts = testset.to_pandas().get("contexts")
                if contexts is not None:
                    context = contexts[0][0]
                else:
                    self._logger.warn(f"TID: {tid}, Testset's context is None!, testset: {testset}")
                    continue
                seed_example_model = SeedExamplesModel(context=context, questions_and_answers=[])
                for _, row in testset.to_pandas().iterrows():
                    question = row.get("question")
                    ground_truth = row.get("ground_truth")
                    metadata = dict(row.get("metadata", [{}])[0])
                    contexts = row.get("contexts")

                    if "status" in metadata.keys():
                        metadata.pop("status")
                    metadata["domain"] = request.domain
                    metadata["referenceContexts"] = contexts
                    metadata["generatorLLM"] = target_model
                    metadata["criticLLM"] = self.critic_llm.get("name")
                    metadata["timestamp"] = datetime.now().timestamp()

                    testset_results.append(
                        TestSetResultModel(
                            question=question, ground_truth=ground_truth, metadata=metadata
                        )
                    )

                    testset_documents.append({
                        "question": question,
                        "groundTruth": ground_truth,
                        "metadata": { **metadata }
                    })

                    seed_example_model.questions_and_answers.append(QnaModel(question=question, answer=ground_truth))
                qna_yaml_model.seed_examples.append(seed_example_model)
        except Exception as e:
            self._logger.error(f"TID: {tid}, {e}", exc_info=True)
            raise OSSRagException(ErrorCode.EVALUATION_TESTSET_RESULT_EXCEPTION, e) from e

        return testset_results, testset_documents, qna_yaml_model
    
    def _get_target_model(self, tid: str, request: RequestModel) -> str:
        try:
            target_model = (
                request.target_model
                if request.target_model is not None
                else json.loads(self.llm.models())["data"][0]["id"]
            )
        except Exception as e:
            self._logger.error(f"TID: {tid}, {e}", exc_info=True)
            raise OSSRagException(ErrorCode.EVALUATION_GENERATOR_LLM_NOT_FOUND, e) from e

        return target_model

    def _make_evaluation_search_query(self, domain: str) -> Dict:
        query_dsl = {}

        query_dsl["query"] = {
            "bool": {
                "must": [
                    {
                        "match": {
                            "metadata.domain": domain
                        }
                    }
                ]
            }
        }

        query_dsl["size"] = 10000

        return query_dsl
    
    def _get_target_evaluation_documents(self, tid: str, request: EvaluationRequestModel) -> List:
        testset_index_name = (
            request.testset_index_name
            if request.testset_index_name is not None
            else self.testset_index_name
        )
        
        query_dsl = self._make_evaluation_search_query(request.domain)

        target_documents = self.vector_store.search(tid=tid, index_name=testset_index_name, body=query_dsl)

        return target_documents
    
    def evaluate(self, tid: str, request: EvaluationRequestModel) -> EvaluationResponseModel:
        target_model = self._get_target_model(tid, request)
        target_documents = self._get_target_evaluation_documents(tid, request)
        dataset = self._generate_dataset(tid=tid, documents=target_documents, target_model=target_model, k=request.k)
        testset_results, testset_documents = self._get_evaluation_result(tid=tid, dataset=dataset, documents=target_documents, target_model=target_model)

        evaluation_index_name = (
            request.evaluation_index_name
            if request.evaluation_index_name is not None
            else self.evaluation_index_name
        )

        self.vector_store.bulk(tid=tid, index_name=evaluation_index_name, documents=testset_documents)

        return EvaluationResponseModel(
            status="ok",
            message=EvaluationMessageModel(
                data=testset_results,
                target_model=target_model,
                critic_model=self.critic_llm["name"],
            ),
        )
    
    def _generate_dataset(
        self, tid: str, documents: List, target_model: str, k: int
    ) -> Dataset:
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        
        system_prompt, user_prompt = self._get_prompts(self._system_prompt, "RAG")
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"], template=user_prompt
        )

        try:
            for document in documents:
                data["question"].append(document["question"])
                data["ground_truth"].append(document["groundTruth"])
                search_results = [
                    docs.page_content
                    for docs in getattr(
                        self.vector_store,
                        (
                            "hybrid_search"
                            if self.retrieval_type == "hybrid"
                            else (
                                "lexical_search"
                                if self.retrieval_type == "lexical"
                                else "semantic_search"
                            )
                        ),
                    )(document["question"], size=k)
                ]
                prompt_str = prompt_template.format(
                    context=search_results, question=document["question"]
                )
                llm_request = OpenAIRequestModel(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_str},
                    ],
                    model=target_model,
                )
                response = json.loads(self.llm.chat_completions(llm_request))
                data["answer"].append(response["choices"][0]["message"]["content"])
                data["contexts"].append(search_results)
        except Exception as e:
            self._logger.error(f"TID: {tid}, {e}", exc_info=True)
            raise OSSRagException(ErrorCode.EVALUATION_DATASET_GENERATION_FAIL, e) from e

        return Dataset.from_dict(data)
    
    @staticmethod
    def _get_prompts(
        system_prompt: str = DEFAULT_SYSTEM_PROMPT, type: str = None
    ) -> tuple[str, str]:
        if type == "RAG":
            system_prompt = f"{system_prompt}\n{EXTRA_RAG_SYSTEM_PROMPT}"
        elif type == "SUMMARY":
            system_prompt = SUMMARY_SYSTEM_PROMPT
        
        user_prompt = """
Question: {question}
Context: {context}
Answer:
"""
        return system_prompt, user_prompt
    @staticmethod
    def _filter_nonfloat(value: Any) -> float:
        if np.isnan(value):
            return -1.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return -1.0
    
    def _get_evaluation_result(
        self, tid: str, dataset: Dataset, documents: List, target_model: str
    ) -> List[EvaluationResultModel]:
        system_prompt, _ = self._get_prompts(self._system_prompt)
        result = evaluate(
            dataset=dataset,
            llm=CompletionLLM(self.critic_llm, system_prompt),
            embeddings=self.embedding,
            metrics=self.metric_modules,
            raise_exceptions=False,
        )
        dataset_dict = result.dataset.to_dict()
        scores_dict = result.scores.to_dict()
        self._logger.info(f"Dataset: {json.dumps(dataset_dict)}")
        self._logger.info(f"Scores: {json.dumps(scores_dict)}")

        testset_results = []
        testset_documents = []
        try:
            for i in range(len(dataset_dict["question"])):
                testset_result, testset_document = self._generate_evaluation_result(dataset_dict, scores_dict, documents, target_model, i)
                testset_results.append(testset_result)
                testset_documents.append(testset_document)
        except Exception as e:
            self._logger.error(f"TID: {tid}, {e}", exc_info=True)
            raise OSSRagException(ErrorCode.EVALUATION_RESULT_GENERATION_FAIL, e) from e
        
        return testset_results, testset_documents
    
    def _generate_evaluation_result(
        self,
        dataset_dict: Dict[str, List],
        scores_dict: Dict[str, List],
        documents: List,
        target_model: str,
        index: int,
    ) -> EvaluationResultModel:
        question = dataset_dict["question"][index]
        ground_truth = dataset_dict["ground_truth"][index]
        metadata = []
        for document in documents:
            if (
                question == document["question"]
                and ground_truth == document["groundTruth"]
            ):
                metadata = document["metadata"]

        metadata["generatorLLM"] = target_model
        metadata["criticLLM"] = self.critic_llm.get("name")
        metadata["timestamp"] = datetime.now().timestamp()

        testset_result = EvaluationResultModel(
            question=question,
            ground_truth=ground_truth,
            answer=dataset_dict["answer"][index],
            contexts=dataset_dict["contexts"][index],
            metadata=metadata,
            scores={
                metric: self._filter_nonfloat(scores_dict[metric][index])
                for metric in self.metric_names
                if metric in scores_dict
            },
        )

        testset_document = {
            "question": testset_result.question,
            "answer": testset_result.answer,
            "contexts": testset_result.contexts,
            "groundTruth": testset_result.ground_truth,
            "metadata": { **testset_result.metadata },
            "scores": {
                "faithfulness": testset_result.scores.get("faithfulness", -1.0) if testset_result.scores is not None else -1.0,
                "answerRelevancy": testset_result.scores.get("answer_relevancy", -1.0) if testset_result.scores is not None else -1.0,
                "contextPrecision": testset_result.scores.get("context_precision", -1.0) if testset_result.scores is not None else -1.0,
                "contextRecall": testset_result.scores.get("context_recall", -1.0) if testset_result.scores is not None else -1.0,
                "contextEntityRecall": testset_result.scores.get("context_entity_recall", -1.0) if testset_result.scores is not None else -1.0,
                "noiseSensitivity": testset_result.scores.get("noise_sensitivity_relevant", -1.0) if testset_result.scores is not None else -1.0,
                "answerSimilarity": testset_result.scores.get("answer_similarity", -1.0) if testset_result.scores is not None else -1.0,
            }
        }

        return testset_result, testset_document
