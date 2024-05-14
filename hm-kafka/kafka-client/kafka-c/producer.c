#include <glib.h>
#include <librdkafka/rdkafka.h>

#include "common.c"

#define ARR_SIZE(arr) (sizeof((arr)) / sizeof((arr[0])))

static void delivery_report(rd_kafka_t *kafka_handle,
                            const rd_kafka_message_t *rkmessage, void *opaque) {
  if (rkmessage->err) {
    g_error("Message delivery failed: %s", rd_kafka_err2str(rkmessage->err));
  }
}

int main(int argc, char **argv) {
  rd_kafka_t *producer;
  rd_kafka_conf_t *conf;
  char errstr[512];

  // Parse the command line
  if (argc != 2) {
    g_error("Usage: %s <config.ini>", argv[0]);
    return 1;
  }

  // Parse the configuration
  // See https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
  const char *config_file = argv[1];

  g_autoptr(GError) error = NULL;
  g_autoptr(GKeyFile) key_file = g_key_file_new();
  if (!g_key_file_load_from_file(key_file, config_file, G_KEY_FILE_NONE,
                                 &error)) {
    g_error("Error loading config file: %s", error->message);
    return 1;
  }

  // Load the relevant configuration sections
  conf = rd_kafka_conf_new();
  load_config_group(conf, key_file, "default");

  // Install a delivery-error callback
  rd_kafka_conf_set_dr_msg_cb(conf, delivery_report);

  // Create the Producer instance
  producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
  if (!producer) {
    g_error("Failed to create new producer: %s", errstr);
    return 1;
  }

  // Configuration object is now owned, and freed, by the rd_kafka_t instance
  conf = NULL;

  // Produce data by selecting random values from these lists
  int message_count = 10;
  const char *topic = "purchases";
  const char *user_ids[6] = {"eabara",   "jsmith",  "sgarcia",
                             "jbernard", "htanaka", "awalther"};
  const char *products[5] = {"book", "alarm clock", "t-shirts", "gift card",
                             "batteries"};

  for (int i = 0; i < message_count; i++) {
    const char *key = user_ids[random() % ARR_SIZE(user_ids)];
    const char *value = products[random() % ARR_SIZE(products)];
    size_t key_len = strlen(key);
    size_t value_len = strlen(value);

    rd_kafka_resp_err_t err;

    err = rd_kafka_producev(producer, RD_KAFKA_V_TOPIC(topic),
                            RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
                            RD_KAFKA_V_KEY((void *)key, key_len),
                            RD_KAFKA_V_VALUE((void *)value, value_len),
                            RD_KAFKA_V_OPAQUE(NULL), RD_KAFKA_V_END);

    if (err) {
      g_error("Failed to produce to topic %s: %s", topic,
              rd_kafka_err2str(err));
      return 1;
    } else {
      g_message("Produced event to topic %s: key = %12s value = %12s", topic,
                key, value);
    }

    rd_kafka_poll(producer, 0);
  }

  // Block until the messages are all sent
  g_message("Flushing final messages..");
  rd_kafka_flush(producer, 10 * 1000);

  if (rd_kafka_outq_len(producer) > 0) {
    g_error("%d message(s) were not delivered", rd_kafka_outq_len(producer));
  }

  g_message("%d events were produced to topic %s.", message_count, topic);
  rd_kafka_destroy(producer);
  return 0;
}
