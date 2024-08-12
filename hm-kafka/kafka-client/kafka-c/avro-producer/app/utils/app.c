#include "app.h"

#include <signal.h>

static void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    is_running = false;
  }
}
