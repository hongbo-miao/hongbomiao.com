#include <glib.h>
#include <librdkafka/rdkafka.h>

#include "config.c"

static volatile sig_atomic_t is_running = 1;

// Signal termination of program
static void stop(int sig) { is_running = 0; }

int main(int argc, char **argv) {
  const char *topic = "production.iot.device.json";

  rd_kafka_t *consumer;
  rd_kafka_conf_t *conf;
  rd_kafka_resp_err_t rd_kafka_resp_err;
  char err_str[4096];

  if (argc != 2) {
    g_error("Usage: %s <config.ini>", argv[0]);
  }

  const char *config_file = argv[1];

  g_autoptr(GError) error = NULL;
  g_autoptr(GKeyFile) key_file = g_key_file_new();
  if (!g_key_file_load_from_file(key_file, config_file, G_KEY_FILE_NONE,
                                 &error)) {
    g_error("Error loading config file: %s", error->message);
  }

  conf = rd_kafka_conf_new();
  load_config_group(conf, key_file, "default");

  consumer = rd_kafka_new(RD_KAFKA_CONSUMER, conf, err_str, sizeof(err_str));
  if (!consumer) {
    g_error("Failed to create new consumer: %s", err_str);
  }
  conf = NULL;
  rd_kafka_poll_set_consumer(consumer);

  // Convert the list of topics to a format suitable for librdkafka
  rd_kafka_topic_partition_list_t *subscription =
      rd_kafka_topic_partition_list_new(1);
  rd_kafka_topic_partition_list_add(subscription, topic, RD_KAFKA_PARTITION_UA);

  // Subscribe to the list of topics
  rd_kafka_resp_err = rd_kafka_subscribe(consumer, subscription);
  if (rd_kafka_resp_err) {
    g_error("Failed to subscribe to %d topics: %s", subscription->cnt,
            rd_kafka_err2str(rd_kafka_resp_err));
  }

  rd_kafka_topic_partition_list_destroy(subscription);

  signal(SIGINT, stop);

  while (is_running) {
    rd_kafka_message_t *consumer_message;

    consumer_message = rd_kafka_consumer_poll(consumer, 500);
    if (!consumer_message) {
      g_message("Waiting...");
      continue;
    }

    if (consumer_message->rd_kafka_resp_err) {
      if (consumer_message->rd_kafka_resp_err ==
          RD_KAFKA_RESP_ERR__PARTITION_EOF) {
        // We can ignore this error - it just means we've read everything and
        // are waiting for more data.
      } else {
        g_message("Consumer error: %s",
                  rd_kafka_message_errstr(consumer_message));
        exit(EXIT_FAILURE);
      }
    } else {
      g_message("Consumed event from topic %s: key = %.*s value = %s",
                rd_kafka_topic_name(consumer_message->rkt),
                (int)consumer_message->key_len, (char *)consumer_message->key,
                (char *)consumer_message->payload);
    }

    rd_kafka_message_destroy(consumer_message);
  }

  g_message("Closing consumer");
  rd_kafka_consumer_close(consumer);
  rd_kafka_destroy(consumer);
  exit(EXIT_SUCCESS);
}
