#ifndef LED_STATUS__
#define LED_STATUS__
/* This include is necessary to include the definition of the type
 * `t_ast_clock_tick` */
#include <asterios.h>

#ifdef KSIM__
void log_message(const char *format, ...);
#else
/* disable all calls to log_message() - inhibit side effects */
#define log_message(...) \
  do {                   \
  } while (0)
#endif

enum led_status { ON, OFF, SET_ON, SET_OFF };

/**
 * Display the LED status for each tick of a clock, with a period of
 * `date_increment`.
 *
 * The LED status are interpreted as follows:
 *
 *  - ON: the LED in ON during the whole interval;
 *  - OFF: the LED is OFF during the whole interval;
 *  - SET_ON: the LED is switched ON in the current interval,
 *    therefore the status of the LED is undefined ([-]);
 *  - SET_OFF: the LED is switched OFF in the current interval,
 *    therefore the status of the LED is undefined ([-]);
 */
void display_status(unsigned int interval_length, t_ast_clock_tick date,
                    unsigned int date_increment, enum led_status status);

#endif /* LED_STATUS__ */
