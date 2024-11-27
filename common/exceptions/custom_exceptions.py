# app/exceptions/custom_exceptions.py

from enum import Enum
class ErrorCode(Enum):
    def __init__(self, error_code: int, message: str):
        self._error_code = error_code
        self._message = message

    @property
    def code(self) -> int:
        return self._error_code

    @property
    def message(self) -> str:
        return self._message
    #error code
    #common 10~
    #vector_store 20~
    #chunker 21~
    #upload api 30~
    #update document api 31~
    #evaluation testset generation api 40~
    #evaluation api 41~
    #common
    INVALID_INPUT_PARAMETER = (10001, "Invalid Input Parameter!")
    #vector exception
    VECTOR_STORE_INDEX_FAIL = (20001, "Index Fail!")
    VECTOR_STORE_INVALID_INPUT_VALUE = (20002, "Invalid Input value!")
    VECTOR_STORE_UNKNOWN = (20999, "VectorStore Process Fail")
    #chunker exception
    CHUNKER_UNKNOWN = (21999, "Chunker Fail!")
    #upload exception
    UPLOAD_DOCUMENT_UNKNOWN = (30999, "Upload Document Fail!")
    #update document exception
    UPDATE_DOCUMENT_UNKNOWN= (31999, "Update Document Fail!")
    #evaluation exception
    EVALUATION_GENERATOR_LLM_NOT_FOUND = (40001, "Generator LLM Model Not Found!")
    EVALUATION_DOCUMENTS_SIZE_EXCEPTION = (40002, "The Number of Documents is less than 5!")
    EVALUATION_TESTSET_SIZE_EXCEPTION = (40003, "The Number of Testset is less than 5!")
    EVALUATION_CONTENT_SUMMARIZE_EXCEPTION = (40004, "Summarize Content Fail!")
    EVALUATION_TESTSET_RESULT_EXCEPTION = (40005, "Testset Result Generation Fail!")
    EVALUATION_TESTSET_GENERATION_UNKNOWN = (40999, "Testset Generation Fail!")
    EVALUATION_DATASET_GENERATION_FAIL = (41001, "Evaluation Dataset Generation Fail!")
    EVALUATION_RESULT_GENERATION_FAIL = (41002, "Evaluation Result Generation Fail!")
    EVALUATION_UNKNOWN = (41999, "Evaluation Fail!")

class OSSRagException(Exception):
    def __init__(self, internal_error: ErrorCode, original_exception: Exception = None, detail: str = None):
        self.internal_error = internal_error
        if original_exception is None:
            self.detail = str(internal_error.message) if detail is None else detail
        else:
            self.detail = str(original_exception)

