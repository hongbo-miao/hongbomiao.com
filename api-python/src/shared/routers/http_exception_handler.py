from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from shared.types.error_detail import ErrorDetail
from shared.types.error_response import ErrorResponse


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.detail,
                type="invalid_request_error",
                code=str(exc.status_code),
            ),
        ).model_dump(),
    )
