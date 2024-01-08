from grafanalib.core import (
    OPS_FORMAT,
    Dashboard,
    GaugePanel,
    GridPos,
    Target,
    TimeSeries,
)

dashboard = Dashboard(
    title="Python generated example dashboard",
    description="Example dashboard using the Random Walk and default Prometheus datasource",
    tags=["example"],
    timezone="browser",
    panels=[
        TimeSeries(
            title="Random Walk",
            dataSource="default",
            targets=[
                Target(
                    datasource="grafana",
                    expr="example",
                ),
            ],
            gridPos=GridPos(h=5, w=10, x=0, y=0),
        ),
        GaugePanel(
            title="Random Walk",
            dataSource="default",
            targets=[
                Target(
                    datasource="grafana",
                    expr="example",
                ),
            ],
            gridPos=GridPos(h=5, w=5, x=10, y=0),
        ),
        TimeSeries(
            title="Prometheus http requests",
            dataSource="prometheus",
            targets=[
                Target(
                    expr="rate(prometheus_http_requests_total[5m])",
                    legendFormat="{{ handler }}",
                    refId="A",
                ),
            ],
            unit=OPS_FORMAT,
            gridPos=GridPos(h=5, w=15, x=0, y=5),
        ),
    ],
).auto_panel_ids()
