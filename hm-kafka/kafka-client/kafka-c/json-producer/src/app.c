#include "../include/app.h"

volatile sig_atomic_t is_running = 1;

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    is_running = 0;
  }
}
