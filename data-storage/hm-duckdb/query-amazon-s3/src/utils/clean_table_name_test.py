from utils.clean_table_name import clean_table_name


class TestCleanTableName:
    def test_input_string_with_only_alphanumeric_characters(self) -> None:
        assert clean_table_name("abc123") == "abc123"

    def test_input_string_with_only_uppercase_alphanumeric_characters(self) -> None:
        assert clean_table_name("ABC123") == "abc123"

    def test_input_string_with_mix_of_alphanumeric_characters_and_underscores(
        self,
    ) -> None:
        assert clean_table_name("abc_123") == "abc_123"

    def test_input_string_with_only_non_alphanumeric_characters(self) -> None:
        assert clean_table_name("abc/123") == "abc_123"
