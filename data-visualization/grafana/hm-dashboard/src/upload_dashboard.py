import importlib.machinery
import json
import logging
import pathlib

import config
import httpx
from grafanalib._gen import DashboardEncoder
from grafanalib.core import Dashboard

logger = logging.getLogger(__name__)


def upload_to_grafana(dashboard: Dashboard) -> None:
    dashboard_json = json.dumps(
        {
            "dashboard": dashboard.to_json_data(),
            "overwrite": True,
            "message": "Updated by grafanlib",
        },
        cls=DashboardEncoder,
    )
    with httpx.Client() as client:
        res = client.post(
            f"{config.GRAFANA_SERVER_URL}/api/dashboards/db",
            content=dashboard_json,
            headers={
                "Authorization": f"Bearer {config.GRAFANA_SERVICE_ACCOUNT_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        logger.info(res.status_code)
        logger.info(res.content)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    module = importlib.machinery.SourceFileLoader(
        "hm_dashboard",
        pathlib.Path(__file__).parent.joinpath("hm.dashboard.py").resolve().as_posix(),
    ).load_module()
    upload_to_grafana(module.dashboard)
