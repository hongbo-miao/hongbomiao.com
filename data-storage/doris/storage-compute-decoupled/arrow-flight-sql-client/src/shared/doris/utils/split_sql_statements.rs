pub fn split_sql_statements(sql_text: &str) -> Vec<String> {
    sql_text
        .split(';')
        .filter_map(|raw_statement| {
            let statement_lines: Vec<&str> = raw_statement
                .lines()
                .filter(|line| !line.trim_start().starts_with("--"))
                .collect();
            let statement = statement_lines.join("\n").trim().to_string();
            if statement.is_empty() {
                None
            } else {
                Some(statement)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::split_sql_statements;

    #[test]
    fn splits_and_strips_comments_and_blank_statements() {
        let sql_text = "\n-- a comment\nselect 1;\n\nselect 2\nfrom foo;\n";
        assert_eq!(
            split_sql_statements(sql_text),
            vec!["select 1".to_string(), "select 2\nfrom foo".to_string()]
        );
    }
}
