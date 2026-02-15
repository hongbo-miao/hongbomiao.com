from grafanalib.core import (
    OPS_FORMAT,
    Dashboard,
    GaugePanel,
    GridPos,
    Target,
    TimeSeries,
)

dashboard = Dashboard(
    title="Python generated example dashboard",  # ty: ignore[unknown-argument]
    description="Example dashboard using the Random Walk and default Prometheus datasource",  # ty: ignore[unknown-argument]
    tags=["example"],  # ty: ignore[unknown-argument]
    timezone="browser",  # ty: ignore[unknown-argument]
    panels=[  # ty: ignore[unknown-argument]
        TimeSeries(
            title="Random Walk",  # ty: ignore[unknown-argument]
            dataSource="default",  # ty: ignore[unknown-argument]
            targets=[  # ty: ignore[unknown-argument]
                Target(
                    datasource="grafana",  # ty: ignore[unknown-argument]
                    expr="example",  # ty: ignore[unknown-argument]
                ),
            ],
            gridPos=GridPos(  # ty: ignore[unknown-argument]
                h=5,  # ty:ignore[unknown-argument]
                w=10,  # ty:ignore[unknown-argument]
                x=0,  # ty:ignore[unknown-argument]
                y=0,  # ty:ignore[unknown-argument]
            ),
        ),
        GaugePanel(
            title="Random Walk",  # ty: ignore[unknown-argument]
            dataSource="default",  # ty: ignore[unknown-argument]
            targets=[  # ty: ignore[unknown-argument]
                Target(
                    datasource="grafana",  # ty: ignore[unknown-argument]
                    expr="example",  # ty: ignore[unknown-argument]
                ),
            ],
            gridPos=GridPos(  # ty: ignore[unknown-argument]
                h=5,  # ty:ignore[unknown-argument]
                w=5,  # ty:ignore[unknown-argument]
                x=10,  # ty:ignore[unknown-argument]
                y=0,  # ty:ignore[unknown-argument]
            ),
        ),
        TimeSeries(
            title="Prometheus http requests",  # ty: ignore[unknown-argument]
            dataSource="prometheus",  # ty: ignore[unknown-argument]
            targets=[  # ty: ignore[unknown-argument]
                Target(
                    expr="rate(prometheus_http_requests_total[5m])",  # ty: ignore[unknown-argument]
                    legendFormat="{{ handler }}",  # ty: ignore[unknown-argument]
                    refId="A",  # ty: ignore[unknown-argument]
                ),
            ],
            unit=OPS_FORMAT,  # ty: ignore[unknown-argument]
            gridPos=GridPos(  # ty: ignore[unknown-argument]
                h=5,  # ty:ignore[unknown-argument]
                w=15,  # ty:ignore[unknown-argument]
                x=0,  # ty:ignore[unknown-argument]
                y=5,  # ty:ignore[unknown-argument]
            ),
        ),
    ],
).auto_panel_ids()
