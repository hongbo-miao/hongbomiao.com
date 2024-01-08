import importlib.machinery
import json
import logging
import pathlib

import config
import requests
from grafanalib._gen import DashboardEncoder
from grafanalib.core import Dashboard


def upload_to_grafana(dashboard: Dashboard) -> None:
    dashboard_json = json.dumps(
        {
            "dashboard": dashboard.to_json_data(),
            "overwrite": True,
            "message": "Updated by grafanlib",
        },
        cls=DashboardEncoder,
    )
    res = requests.post(
        f"{config.GRAFANA_SERVER_URL}/api/dashboards/db",
        data=dashboard_json,
        headers={
            "Authorization": f"Bearer {config.GRAFANA_SERVICE_ACCOUNT_TOKEN}",
            "Content-Type": "application/json",
        },
        verify=True,
    )
    logging.info(res.status_code)
    logging.info(res.content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    module = importlib.machinery.SourceFileLoader(
        "hm_dashboard",
        pathlib.Path(__file__).parent.joinpath("hm.dashboard.py").resolve().as_posix(),
    ).load_module()
    upload_to_grafana(module.dashboard)
