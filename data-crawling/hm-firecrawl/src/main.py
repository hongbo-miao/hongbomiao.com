import logging
from pathlib import Path

from config import config
from firecrawl.firecrawl import FirecrawlApp

logger = logging.getLogger(__name__)


def main() -> None:
    app = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)

    # Scrape a website:
    res = app.scrape_url(
        "https://firecrawl.dev",
        formats=["markdown"],
    )
    filename = "output.md"
    Path(filename).write_text(res.markdown, encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
