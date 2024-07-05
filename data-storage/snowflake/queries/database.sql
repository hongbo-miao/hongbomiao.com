show databases;
show databases history;

alter database MY_DB set data_retention_time_in_days = 30;

show schemas history in MY_DB;

use database MY_DB;
show parameters in database;
