package com.hongbomiao.sources;

import org.apache.flink.connector.datagen.source.GeneratorFunction;
import org.apache.flink.types.Row;

/**
 * Generates live service-incident state, one component at a time cycling through {@code
 * operational} -> {@code degraded} -> {@code outage} -> {@code operational}. This is world state
 * that changes independently of the ticket stream, so a ticket lookup-joined against it can land
 * mid-outage.
 */
public class ServiceIncidentSource implements GeneratorFunction<Long, Row> {

  private static final String[] COMPONENTS = {
    "billing", "authentication", "api", "mobile", "webhooks", "account"
  };

  private static final String[] STATUSES = {"operational", "degraded", "outage"};

  private static final String[] SUMMARIES = {
    "no known issues",
    "elevated latency, investigating",
    "full outage, engineering paging"
  };

  @Override
  public Row map(Long sequenceNumber) {
    int componentIndex = (int) (sequenceNumber % COMPONENTS.length);
    int statusIndex = (int) ((sequenceNumber / COMPONENTS.length) % STATUSES.length);
    return Row.of(
        COMPONENTS[componentIndex],
        STATUSES[statusIndex],
        SUMMARIES[statusIndex],
        System.currentTimeMillis());
  }
}
