import logging

import sentry_sdk
from config import Settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import health, motor, seed
from sentry_sdk.integrations.fastapi import FastApiIntegration
from utils.logger import logger

settings = Settings()

logger.setLevel(logging.INFO)
logger.info(f"{settings = }")

sentry_sdk.init(
    dsn=settings.SENTRY_DSN,
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
    environment=settings.ENV,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(motor.router)
app.include_router(seed.router)
