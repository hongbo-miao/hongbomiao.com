use crate::shared::parallel_calculation::types::calculation_response::CalculationResponse;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::time::Instant;

pub fn calculate_parallel(items_count: usize) -> CalculationResponse {
    let start_time = Instant::now();

    // Simulate a heavy calculation: sum of (i^2 * sin(i) * cos(i))
    let result: f64 = (0..items_count)
        .into_par_iter()
        .map(|i| {
            let float_i = i as f64;
            // Simulate expensive computation
            let mut value = float_i.powi(2);
            for _ in 0..1000 {
                value = (value.sin() * value.cos()).abs() + float_i.sqrt();
            }
            value
        })
        .sum();

    let duration = start_time.elapsed();
    let thread_count = rayon::current_num_threads();

    CalculationResponse {
        result,
        duration_milliseconds: duration.as_millis() as u64,
        items_processed: items_count as i32,
        thread_count: thread_count as i32,
    }
}
