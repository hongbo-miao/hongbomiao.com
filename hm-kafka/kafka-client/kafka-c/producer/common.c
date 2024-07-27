#include <glib.h>

static void load_config_group(rd_kafka_conf_t *conf, GKeyFile *key_file,
                              const char *group) {
  char errstr[512];
  g_autoptr(GError) error = NULL;

  gchar **ptr = g_key_file_get_keys(key_file, group, NULL, &error);
  if (error) {
    g_error("%s", error->message);
    exit(1);
  }

  while (*ptr) {
    const char *key = *ptr;
    g_autofree gchar *value =
        g_key_file_get_string(key_file, group, key, &error);

    if (error) {
      g_error("Reading key: %s", error->message);
      exit(1);
    }

    if (rd_kafka_conf_set(conf, key, value, errstr, sizeof(errstr)) !=
        RD_KAFKA_CONF_OK) {
      g_error("%s", errstr);
      exit(1);
    }

    ptr++;
  }
}
