import { MeterProvider } from '@opentelemetry/sdk-metrics-base';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';

const serviceName = 'hm-api-node-metric-service';
const metricProvider = new MeterProvider({
  [SemanticResourceAttributes.SERVICE_NAME]: serviceName,
});

const meter = metricProvider.getMeter('hm-api-node-meter');

export default meter;
