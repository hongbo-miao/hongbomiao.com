import asyncio
import logging
from pathlib import Path

from crawl4ai import AsyncWebCrawler

logger = logging.getLogger(__name__)


async def main() -> None:
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.nytimes.com/section/business",
        )
        filename = "output.md"
        Path(filename).write_text(result.markdown, encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
