#ifndef KAFKA_H
#define KAFKA_H

#include <glib.h>
#include <librdkafka/rdkafka.h>

void delivery_report(rd_kafka_t *kafka_handle, const rd_kafka_message_t *rkmessage, void *opaque);

#endif
