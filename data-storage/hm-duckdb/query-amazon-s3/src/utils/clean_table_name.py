import re


def clean_table_name(s: str) -> str:
    # Lowercase
    s = s.lower()
    # Replace non-alphanumeric characters with _
    return re.sub(r"[^a-z0-9]", "_", s)
