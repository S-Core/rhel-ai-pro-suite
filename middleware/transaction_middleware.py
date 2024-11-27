import uuid
from util import LogUtility
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class TransactionIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger=LogUtility.getLogger()
        # Transaction ID 생성
        transaction_id = str(uuid.uuid4())
        
        # 요청에 transaction ID 속성 추가
        request.state.transaction_id = transaction_id
        
        # 요청 로그 남기기
        logger.info(f"TID: {transaction_id} - Request: {request.method} {request.url}")
        
        # 응답 처리
        response = await call_next(request)
        
        # 응답 로그 남기기
        logger.info(f"TID: {transaction_id} - Response status: {response.status_code}")
        
        # 응답 헤더에 transaction ID 추가
        response.headers["X-Transaction-ID"] = transaction_id
        return response