#include "../include/config.h"

void load_config_group(rd_kafka_conf_t *conf, GKeyFile *key_file, const char *group_name) {
  char err_str[4096];
  g_autoptr(GError) err = NULL;

  gchar **ptr = g_key_file_get_keys(key_file, group_name, NULL, &err);
  if (err) {
    g_error("%s", err->message);
  }

  while (*ptr) {
    const char *key = *ptr;
    g_autofree gchar *value = g_key_file_get_string(key_file, group_name, key, &err);
    if (err) {
      g_error("Reading key: %s", err->message);
    }
    if (rd_kafka_conf_set(conf, key, value, err_str, sizeof(err_str)) != RD_KAFKA_CONF_OK) {
      g_error("%s", err_str);
    }
    ptr++;
  }
}
