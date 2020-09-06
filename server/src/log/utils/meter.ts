import { CollectorMetricExporter } from '@opentelemetry/exporter-collector';
import { MeterProvider } from '@opentelemetry/metrics';

const metricExporter = new CollectorMetricExporter({
  serviceName: 'server-metric-service',
});

const meter = new MeterProvider({
  exporter: metricExporter,
  interval: 60 * 1000, // 60 seconds
}).getMeter('server-meter');

export default meter;
