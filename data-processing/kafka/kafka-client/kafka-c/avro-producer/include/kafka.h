#ifndef INCLUDE_KAFKA_H_
#define INCLUDE_KAFKA_H_

#include <glib.h>
#include <librdkafka/rdkafka.h>

void delivery_report(rd_kafka_t *kafka_handle, const rd_kafka_message_t *rkmessage, void *opaque);

#endif  // INCLUDE_KAFKA_H_
