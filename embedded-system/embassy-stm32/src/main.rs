#![no_std]
#![no_main]

use core::fmt::Write as _;

use embassy_executor::Spawner;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_stm32::usart::{Config as UsartConfig, Uart};
use embassy_time::Timer;
use heapless::String;
use panic_halt as _;

#[embassy_executor::task]
async fn blink_led(mut led: Output<'static>) {
    loop {
        led.toggle();
        Timer::after_millis(500).await;
    }
}

#[embassy_executor::task]
async fn report_uptime(mut uart: Uart<'static, embassy_stm32::mode::Blocking>) {
    let mut second_count: u32 = 0;
    loop {
        Timer::after_secs(1).await;
        second_count += 1;

        let mut message: String<32> = String::new();
        write!(message, "Uptime: {second_count} second(s)\r\n")
            .expect("Failed to format uptime message");
        uart.blocking_write(message.as_bytes())
            .expect("Failed to write uptime message to USART2");
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let peripherals = embassy_stm32::init(Default::default());

    let led = Output::new(peripherals.PD12, Level::Low, Speed::Low);
    let uart = Uart::new_blocking(
        peripherals.USART2,
        peripherals.PA3,
        peripherals.PA2,
        UsartConfig::default(),
    )
    .expect("Failed to initialize USART2");

    spawner.spawn(blink_led(led).expect("Failed to spawn blink LED task"));
    spawner.spawn(report_uptime(uart).expect("Failed to spawn report uptime task"));
}
