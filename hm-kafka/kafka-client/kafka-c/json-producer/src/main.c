#include <cjson/cJSON.h>
#include <glib.h>
#include <librdkafka/rdkafka.h>
#include <signal.h>
#include <stdlib.h>

#include "../include/app.h"
#include "../include/config.h"
#include "../include/kafka.h"

int main(int argc, char **argv) {
  const char *topic = "production.iot.device.json";

  rd_kafka_t *producer;
  rd_kafka_conf_t *conf;
  char err_str[4096];

  if (argc != 2) {
    g_error("Usage: %s <config.ini>", argv[0]);
  }

  const char *config_file = argv[1];
  g_autoptr(GError) err = NULL;
  g_autoptr(GKeyFile) key_file = g_key_file_new();
  if (!g_key_file_load_from_file(key_file, config_file, G_KEY_FILE_NONE, &err)) {
    g_error("Error loading config file: %s", err->message);
  }

  conf = rd_kafka_conf_new();
  load_config_group(conf, key_file, "default");
  rd_kafka_conf_set_dr_msg_cb(conf, delivery_report);
  producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, err_str, sizeof(err_str));
  if (!producer) {
    g_error("Failed to create new producer: %s", err_str);
  }
  conf = NULL;

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  const char *device_ids[5] = {"device1", "device2", "device3", "device4", "device5"};
  const char *status_list[3] = {"online", "offline", "maintenance"};
  const char *locations[3] = {"locationA", "locationB", "locationC"};
  const char *types[3] = {"type1", "type2", "type3"};

  while (is_running) {
    const char *kafka_key = device_ids[random() % G_N_ELEMENTS(device_ids)];

    const char *status = status_list[random() % G_N_ELEMENTS(status_list)];
    const char *location = locations[random() % G_N_ELEMENTS(locations)];
    const char *type = types[random() % G_N_ELEMENTS(types)];
    double temperature = ((double)random() / RAND_MAX) * 100.0 - 50.0;  // [-50.0, 50.0]
    double humidity = (random() % 101) / 100.0;                         // [0.0, 1.0]
    int battery = random() % 101;                                       // [0, 100]
    int signal_strength = random() % 101;                               // [0, 100]
    const char *mode = (random() % 2) ? "manual" : "auto";
    bool active = (random() % 2) ? true : false;

    // Create JSON object
    cJSON *json_obj = cJSON_CreateObject();
    if (!json_obj) {
      g_error("Failed to create JSON object");
    }
    cJSON_AddStringToObject(json_obj, "status", status);
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
      g_error("Failed to print JSON object\n");
    }

    size_t key_len = strlen(kafka_key);
    rd_kafka_resp_err_t rd_kafka_resp_err;
    rd_kafka_resp_err =
      rd_kafka_producev(producer, RD_KAFKA_V_TOPIC(topic), RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
                        RD_KAFKA_V_KEY((void *)kafka_key, key_len),
                        RD_KAFKA_V_VALUE((void *)json_str, strlen(json_str)),
                        RD_KAFKA_V_OPAQUE(NULL), RD_KAFKA_V_END);

    if (rd_kafka_resp_err) {
      g_error("Failed to produce to topic %s: %s", topic, rd_kafka_err2str(rd_kafka_resp_err));
    }

    cJSON_Delete(json_obj);
    free((void *)json_str);

    rd_kafka_poll(producer, 0);
    g_usleep(50);  // μs
  }

  g_message("Flushing final messages ...");
  rd_kafka_flush(producer, 10 * 1000);

  if (rd_kafka_outq_len(producer) > 0) {
    g_error("%d message(s) were not delivered", rd_kafka_outq_len(producer));
  }

  g_message("Producer stopped.");
  rd_kafka_destroy(producer);
  exit(EXIT_SUCCESS);
}
