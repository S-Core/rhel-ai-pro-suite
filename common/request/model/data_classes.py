from enum import Enum
from pydantic import BaseModel, constr
from typing import Optional, List, Dict
from fastapi import Form

class LLMRequestModel(BaseModel):
    prompt: str
    model: str
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n_keep: Optional[int] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = None

class Message(BaseModel):
    role: str
    content: str

class AllowedDocumentStatus(Enum):
    TRAINED = "trained"

class DoucmentFilterModel(BaseModel):
    """
    Represents a filter to search for documents by domain.

    Attributes:
        domain (str): The domain of the document to filter by. It must be a non-empty string
        without leading or trailing whitespace.
    """
    domain: constr(strip_whitespace=True, min_length=1) # type: ignore

class UpdateDocumentModel(BaseModel):
    """
    Represents the update data for a document.

    Attributes:
        status (AllowedDocumentStatus): The status of the document to be updated.
    """
    status: AllowedDocumentStatus
    class Config:
        use_enum_values = True

class UpdateDocumentRequestModel(BaseModel):
    """
    Represents the request to update a document.

    This model contains both the filter to identify the document and the update data.

    Attributes:
        filter (DocumentFilterModel): The filter to search for the document by domain.
        update (UpdateDocumentModel): The update data for the document.
    """
    filter: DoucmentFilterModel
    update: UpdateDocumentModel

class DocumentMetaDataModel(BaseModel):
    source: Optional[str] = None
    name: Optional[str] = None
    domain: constr(strip_whitespace=True, min_length=1) # type: ignore
    author: Optional[str] = None

def parse_document_meta_data_model(
    source: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    domain: constr(strip_whitespace=True, min_length=1) = Form(...),  # type: ignore
    author: Optional[str] = Form(None)
) -> DocumentMetaDataModel:
    return DocumentMetaDataModel(source=source, name=name, domain=domain, author=author)

class DocumentModel(BaseModel):
    content: constr(strip_whitespace=True, min_length=1) # type: ignore
    metadata: DocumentMetaDataModel
    index_name: Optional[str] = None

class OpenAIRequestModel(BaseModel):
    messages: List[Message]
    model: str
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[object] = None
    seed: Optional[int] = None
    stop: Optional[str | List] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

class RerankRequestModel(BaseModel):
    question: str
    contexts: List[str]
    k: int = 3
