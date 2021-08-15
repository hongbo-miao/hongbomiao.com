import { CollectorMetricExporter } from '@opentelemetry/exporter-collector';
import { MeterProvider } from '@opentelemetry/metrics';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import isDevelopment from '../../shared/utils/isDevelopment';

const serviceName = 'hm-api-node-metric-service';
const metricExporter = new CollectorMetricExporter();
const metricProvider = isDevelopment()
  ? new MeterProvider({
      [SemanticResourceAttributes.SERVICE_NAME]: serviceName,
      exporter: metricExporter,
      interval: 5e3, // 5s
    })
  : new MeterProvider({
      [SemanticResourceAttributes.SERVICE_NAME]: serviceName,
    });
const meter = metricProvider.getMeter('hm-api-node-meter');

export default meter;
