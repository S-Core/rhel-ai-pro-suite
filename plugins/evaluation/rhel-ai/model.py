import json
from pydantic import BaseModel, validator
from typing import Optional, List, Dict


class CriticLLM(BaseModel):
    host: str
    model: str
    headers: Optional[Dict] = {}


class RequestModel(BaseModel):
    target_model: Optional[str]
    critic_llm: Optional[CriticLLM]


class QnaYamlRequestModel(BaseModel):
    version: Optional[int] = 3
    created_by: str
    document_outline: Optional[str]
    repo: str
    commit: str
    patterns: List[str]

    @validator("version", pre=True, always=True)
    def set_default_testset_size(cls, v):
        return v or 3

    @validator("version")
    def check_testset_size(cls, v):
        if v < 3:
            raise ValueError("version must be at least 3")
        return v


class TestSetRequestModel(RequestModel):
    testset_size: Optional[int] = 3
    domain: str
    document_index_name: Optional[str]
    testset_index_name: Optional[str]
    qna_yaml: QnaYamlRequestModel

    @validator("testset_size", pre=True, always=True)
    def set_default_testset_size(cls, v):
        return v or 3

    @validator("testset_size")
    def check_testset_size(cls, v):
        if v < 3:
            raise ValueError("testset_size must be at least 3")
        return v


class TestSetResultModel(BaseModel):
    question: str
    ground_truth: str
    metadata: Optional[Dict]


class TestSetMessageModel(BaseModel):
    data: List[TestSetResultModel]
    target_model: str
    critic_model: str


class TestSetResponseModel(BaseModel):
    status: str
    message: str | TestSetMessageModel


class EvaluationRequestModel(RequestModel):
    k: Optional[int] = 3
    domain: str
    testset_index_name: Optional[str]
    evaluation_index_name: Optional[str]

    @validator("k", pre=True, always=True)
    def set_default_testset_size(cls, v):
        return v or 3


class EvaluationResultModel(BaseModel):
    question: str
    ground_truth: str
    answer: Optional[str]
    contexts: Optional[List[str]]
    metadata: Optional[Dict]
    scores: Dict


class EvaluationMessageModel(BaseModel):
    data: List[EvaluationResultModel]
    target_model: str
    critic_model: str


class EvaluationResponseModel(BaseModel):
    status: str
    message: str | EvaluationMessageModel

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class QnaModel(BaseModel):
    question: str
    answer: str


class SeedExamplesModel(BaseModel):
    context: Optional[str]
    questions_and_answers: List[QnaModel]


class DocumentModel(BaseModel):
    repo: str
    commit: str
    patterns: List[str]


class QnaYamlModel(BaseModel):
    version: int
    created_by: str
    domain: str
    seed_examples: List[SeedExamplesModel]
    document_outline: Optional[str]
    document: DocumentModel
