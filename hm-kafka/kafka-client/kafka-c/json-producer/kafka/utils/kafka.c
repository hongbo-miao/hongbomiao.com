#include <glib.h>
#include <librdkafka/rdkafka.h>

static void delivery_report(rd_kafka_t *kafka_handle,
                            const rd_kafka_message_t *rkmessage, void *opaque) {
  if (rkmessage->err) {
    g_error("Message delivery failed: %s", rd_kafka_err2str(rkmessage->err));
  }
}
