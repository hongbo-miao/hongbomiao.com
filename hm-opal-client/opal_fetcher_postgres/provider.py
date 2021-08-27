# https://github.com/authorizon/opal-fetcher-postgres

import json
from typing import List, Optional

import asyncpg
from asyncpg.exceptions import DataError
from asyncpg.transaction import Transaction
from opal_common.fetcher.events import FetcherConfig, FetchEvent
from opal_common.fetcher.fetch_provider import BaseFetchProvider
from opal_common.logger import logger
from pydantic import BaseModel, Field
from tenacity import retry_unless_exception_type, stop, wait


class PostgresConnectionParams(BaseModel):
    database: Optional[str] = Field(None, description="PostgreSQL database")
    user: Optional[str] = Field(None, description="PostgreSQL user")
    password: Optional[str] = Field(None, description="PostgreSQL password")
    host: Optional[str] = Field(None, description="PostgreSQL host")
    port: Optional[str] = Field(None, description="PostgreSQL port")


class PostgresFetcherConfig(FetcherConfig):
    fetcher: str = "PostgresFetchProvider"
    connection_params: Optional[PostgresConnectionParams] = Field(
        None, description="can be overridden or complement parts of the DSN"
    )
    query: str = Field(..., description="the query")
    fetch_one: bool = Field(False, description="fetch only one row")
    dictKey: Optional[str] = Field(
        None, description="array of dict will map to dict with provided dictKey"
    )


class PostgresFetchEvent(FetchEvent):
    fetcher: str = "PostgresFetchProvider"
    config: Optional[PostgresFetcherConfig] = None


class PostgresFetchProvider(BaseFetchProvider):
    RETRY_CONFIG = {
        "wait": wait.wait_random_exponential(),
        "stop": stop.stop_after_attempt(10),
        "retry": retry_unless_exception_type(
            DataError
        ),  # query error (i.e: invalid table, etc)
        "reraise": True,
    }

    def __init__(self, event: PostgresFetchEvent) -> None:
        if event.config is None:
            event.config = PostgresFetcherConfig()
        super().__init__(event)
        self._connection: Optional[asyncpg.Connection] = None
        self._transaction: Optional[Transaction] = None

    def parse_event(self, event: FetchEvent) -> PostgresFetchEvent:
        return PostgresFetchEvent(**event.dict(exclude={"config"}), config=event.config)

    async def __aenter__(self):
        # self._event: PostgresFetchEvent  # type casting

        dsn: str = self._event.url
        connection_params: dict = (
            {}
            if self._event.config.connection_params is None
            else self._event.config.connection_params.dict(exclude_none=True)
        )

        self._connection: asyncpg.Connection = await asyncpg.connect(
            dsn, **connection_params
        )

        await self._connection.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )

        self._transaction: Transaction = self._connection.transaction(readonly=True)
        await self._transaction.__aenter__()

        return self

    async def __aexit__(self, exc_type=None, exc_val=None, tb=None):
        if self._transaction is not None:
            await self._transaction.__aexit__(exc_type, exc_val, tb)
        if self._connection is not None:
            await self._connection.close()

    async def _fetch_(self):
        # self._event: PostgresFetchEvent  # type casting

        if self._event.config is None:
            logger.warning("Incomplete fetcher config!")
            return

        logger.debug(f"{self.__class__.__name__} fetching from {self._url}")

        if self._event.config.fetch_one:
            row = await self._connection.fetchrow(self._event.config.query)
            return [row]
        else:
            return await self._connection.fetch(self._event.config.query)

    async def _process_(self, records: List[asyncpg.Record]):
        # self._event: PostgresFetchEvent  # type casting

        if self._event.config is not None and self._event.config.fetch_one:
            if records and len(records) > 0:
                return dict(records[0])
            return {}
        else:
            if self._event.config.dictKey is None:
                return [dict(record) for record in records]
            return {
                dict(record)[self._event.config.dictKey]: dict(record)
                for record in records
            }
