# embassy-stm32

A minimal `embassy-stm32` firmware that runs without any physical hardware, using the Renode simulator.
It spawns two `async fn` tasks onto embassy's executor: one blinks an LED every 500 ms, the other writes an incrementing uptime counter over USART2 every 1 s.

There is one CPU core, no OS, and no RTOS. Each task's only blocking-looking call is `Timer::after_*(...).await`, which yields control back to the executor instead of halting the CPU, letting the other task run.
That is the whole concurrency mechanism: cooperative yielding at `.await` points, not OS thread preemption and not an RTOS scheduler.
A task that never awaits would starve the other one permanently.
