#include <cjson/cJSON.h>
#include <glib.h>
#include <librdkafka/rdkafka.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>

#include "common.c"

#define ARR_SIZE(arr) (sizeof((arr)) / sizeof((arr[0])))

// Global variable to manage graceful shutdown
static volatile bool is_running = true;

static void delivery_report(rd_kafka_t *kafka_handle,
                            const rd_kafka_message_t *rkmessage, void *opaque) {
  if (rkmessage->err) {
    g_error("Message delivery failed: %s", rd_kafka_err2str(rkmessage->err));
  }
}

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    is_running = false;
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
  // https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
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

  // Increase the internal queue buffer size
  rd_kafka_conf_set(conf, "queue.buffering.max.messages", "10000000", NULL, 0);

  rd_kafka_conf_set(conf, "queue.buffering.max.kbytes",
                    "10485760",  // 10 GiB
                    NULL, 0, );

  // Adjust batch size and linger time
  rd_kafka_conf_set(conf, "batch.size", "65536", NULL, 0);  // 64 KiB
  rd_kafka_conf_set(conf, "linger.ms", "5", NULL, 0);       // 5 ms

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

  // Set up signal handling for graceful shutdown
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  // Produce data continuously
  const char *topic = "production.iot.device.json";
  const char *device_ids[6] = {"device1", "device2", "device3",
                               "device4", "device5", "device6"};
  const char *status[3] = {"online", "offline", "maintenance"};
  const char *locations[3] = {"locationA", "locationB", "locationC"};
  const char *types[3] = {"type1", "type2", "type3"};

  while (is_running) {
    const char *key = device_ids[random() % ARR_SIZE(device_ids)];
    const char *status_value = status[random() % ARR_SIZE(status)];
    const char *location = locations[random() % ARR_SIZE(locations)];
    const char *type = types[random() % ARR_SIZE(types)];
    int temperature = (random() % 100) - 50;     // Range from -50 to 49
    double humidity = (random() % 101) / 100.0;  // Range from 0.0 to 1.0
    int battery = random() % 101;                // Range from 0 to 100
    int signal_strength = random() % 101;        // Range from 0 to 100
    const char *mode = (random() % 2) ? "manual" : "auto";
    bool active = (random() % 2) ? true : false;

    // Create JSON object
    cJSON *json_obj = cJSON_CreateObject();
    if (!json_obj) {
      g_error("Failed to create JSON object");
      break;
    }
    cJSON_AddStringToObject(json_obj, "status", status_value);
    cJSON_AddStringToObject(json_obj, "location", location);
    cJSON_AddStringToObject(json_obj, "type", type);
    cJSON_AddNumberToObject(json_obj, "temperature", temperature);
    cJSON_AddNumberToObject(json_obj, "humidity", humidity);
    cJSON_AddNumberToObject(json_obj, "battery", battery);
    cJSON_AddNumberToObject(json_obj, "signal_strength", signal_strength);
    cJSON_AddStringToObject(json_obj, "mode", mode);
    cJSON_AddBoolToObject(json_obj, "active", active);

    // Convert JSON object to string
    const char *json_str = cJSON_PrintUnformatted(json_obj);
    if (!json_str) {
      g_error("Failed to print JSON object");
      cJSON_Delete(json_obj);
      break;
    }

    size_t key_len = strlen(key);
    size_t value_len = strlen(json_str);

    rd_kafka_resp_err_t err;

    err = rd_kafka_producev(producer, RD_KAFKA_V_TOPIC(topic),
                            RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
                            RD_KAFKA_V_KEY((void *)key, key_len),
                            RD_KAFKA_V_VALUE((void *)json_str, value_len),
                            RD_KAFKA_V_OPAQUE(NULL), RD_KAFKA_V_END);

    if (err) {
      g_error("Failed to produce to topic %s: %s", topic,
              rd_kafka_err2str(err));
      return 1;
    }

    // Free the JSON object
    cJSON_Delete(json_obj);
    free((void *)json_str);  // cJSON_PrintUnformatted allocates memory

    rd_kafka_poll(producer, 0);
    g_usleep(50);  // Sleep for 10 us
  }

  // Block until the messages are all sent
  g_message("Flushing final messages ...");
  rd_kafka_flush(producer, 10 * 1000);

  if (rd_kafka_outq_len(producer) > 0) {
    g_error("%d message(s) were not delivered", rd_kafka_outq_len(producer));
  }

  g_message("Producer stopped.");
  rd_kafka_destroy(producer);
  return 0;
}
