#ifndef APP_H
#define APP_H

#include <signal.h>

extern volatile sig_atomic_t is_running;

void signal_handler(int signal);

#endif
