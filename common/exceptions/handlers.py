# app/exceptions/handlers.py
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from .custom_exceptions import ErrorCode, OSSRagException
from util import LogUtility

def error_handler(request: Request,exc: Exception):
    logger=LogUtility.getLogger()
    logger.error("TID : %s, Exception occurred: %s", request.state.transaction_id, exc, exc_info=True)

async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    error_handler(request, exc)
    return JSONResponse(
        status_code=422,
        content={
            "TID": request.state.transaction_id,
            "error code": ErrorCode.INVALID_INPUT_PARAMETER.code,
            "message": ErrorCode.INVALID_INPUT_PARAMETER.message,
            "details": str(exc)
        },
    )

async def oss_rag_exception_handler(request: Request, exc: OSSRagException):
    error_handler(request, exc)
    return JSONResponse(
        status_code=500,
        content={
            "TID": request.state.transaction_id,
            "error code": exc.internal_error.code,
            "message": exc.internal_error.message,
            "detail": exc.detail,
        },
    )

async def default_exception_handler(request: Request, exc: Exception):
    error_handler(request, exc)
    return JSONResponse(
        status_code=500,
        content={
            "TID": request.state.transaction_id,
            "error": "An error occurred while processing request",
            "details": str(exc)
        },
    )

def register_exception_handlers(app):
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
    app.add_exception_handler(OSSRagException, oss_rag_exception_handler)
    app.add_exception_handler(Exception, default_exception_handler)
