pub async fn get_root() -> &'static str {
    "ok"
}

#[cfg(test)]
#[path = "get_root_test.rs"]
mod tests;
