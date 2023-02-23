import "influxdata/influxdb/sample"

sample.data(set: "machineProduction")
  |> to(bucket: "hm-grinding-wheel-station-bucket")
