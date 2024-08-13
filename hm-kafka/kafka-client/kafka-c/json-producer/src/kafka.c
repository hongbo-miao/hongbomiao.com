#include "../include/kafka.h"

void delivery_report(rd_kafka_t *kafka_handle __attribute__((unused)),
                     const rd_kafka_message_t *rkmessage, void *opaque __attribute__((unused))) {
  if (rkmessage->err) {
    g_error("Message delivery failed: %s", rd_kafka_err2str(rkmessage->err));
  }
}
