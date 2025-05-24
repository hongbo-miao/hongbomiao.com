-- Set root password
set password for 'root' = password('xxx');

-- Max column number per table (default: 10000)
admin show frontend config like "max_column_number_per_table";
admin set frontend config ("max_column_number_per_table" = "100000");
