pub async fn root() -> &'static str {
    "ok"
}

#[cfg(test)]
#[path = "root_test.rs"]
mod tests;
