use crate::shared::python_parallel_calculation::types::python_calculation_response::PythonCalculationResponse;
use once_cell::sync::Lazy;
use pyo3::types::{PyAny, PyAnyMethods, PyModule};
use pyo3::{Py, Python};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::ffi::CString;
use std::sync::Mutex;
use std::time::Instant;

static PYTHON_MODULE_CODE: &str = include_str!("../scripts/compute_item.py");

static PYTHON_FUNCTION: Lazy<Mutex<Option<Py<PyAny>>>> = Lazy::new(|| Mutex::new(None));

fn initialize_python_function() -> Result<(), String> {
    let mut function_guard = PYTHON_FUNCTION
        .lock()
        .map_err(|error| format!("Failed to lock mutex: {error}"))?;

    if function_guard.is_none() {
        Python::attach(|python| {
            let code_cstring = CString::new(PYTHON_MODULE_CODE)
                .map_err(|error| format!("Failed to create CString: {error}"))?;
            let file_name_cstring = CString::new("compute_item.py")
                .map_err(|error| format!("Failed to create file name CString: {error}"))?;
            let module_name_cstring = CString::new("compute_item")
                .map_err(|error| format!("Failed to create module name CString: {error}"))?;

            let module = PyModule::from_code(
                python,
                &code_cstring,
                &file_name_cstring,
                &module_name_cstring,
            )
            .map_err(|error| format!("Failed to create Python module: {error}"))?;

            let compute_item_function = module
                .getattr("compute_item")
                .map_err(|error| format!("Failed to get compute_item function: {error}"))?;

            *function_guard = Some(compute_item_function.into());
            Ok(())
        })
    } else {
        Ok(())
    }
}

pub fn calculate_with_python(items_count: i32) -> Result<PythonCalculationResponse, String> {
    initialize_python_function()?;

    let start_time = Instant::now();

    // Use Rayon to parallelize across multiple threads
    // Each thread calls the single-threaded Python compute_item function
    let result: f64 = (0..items_count)
        .into_par_iter()
        .map(|i| {
            // Each Rayon thread acquires the GIL and calls Python
            Python::attach(|python| {
                let function_guard = PYTHON_FUNCTION
                    .lock()
                    .map_err(|error| format!("Failed to lock mutex: {error}"))?;
                let compute_item_function = function_guard
                    .as_ref()
                    .ok_or_else(|| "Python function not initialized".to_string())?;

                compute_item_function
                    .call1(python, (i,))
                    .map_err(|error| format!("Failed to call Python function: {error}"))?
                    .extract::<f64>(python)
                    .map_err(|error| format!("Failed to extract result: {error}"))
            })
        })
        .collect::<Result<Vec<f64>, String>>()?
        .into_iter()
        .sum();

    let duration = start_time.elapsed();
    let thread_count = rayon::current_num_threads();

    Ok(PythonCalculationResponse {
        result,
        duration_milliseconds: duration.as_millis() as u64,
        items_processed: items_count,
        thread_count: thread_count as i32,
    })
}
