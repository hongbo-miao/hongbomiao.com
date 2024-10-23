-- Enable Microsoft SQL Server Agent
-- Sql Server Configuration Manager -> SQL Server Services -> SQL Server Agent - Start Service

-- Create Airbyte user with default database
create login airbyte_user with password = 'xxx', default_database = iot_db;
create user airbyte_user for login airbyte_user;

-- Allow Airbyte to check whether or not the SQL Server Agent is running
use master;
grant view server state to airbyte_user;

use iot_db;

-- Grant read only access
grant select on dbo.battery to airbyte_user;
grant select on dbo.motor to airbyte_user;

-- Enable change data capture (CDC)
-- https://docs.airbyte.com/integrations/sources/mssql/#change-data-capture-cdc

exec sys.sp_cdc_enable_db;
-- Check: select name, is_cdc_enabled from sys.databases where name = 'iot_db';

exec sys.sp_cdc_enable_table @source_schema = N'dbo', @source_name = N'battery', @role_name = null, @supports_net_changes = 0;
exec sys.sp_cdc_enable_table @source_schema = N'dbo', @source_name = N'motor', @role_name = null, @supports_net_changes = 0;

-- Disable
-- exec sys.sp_cdc_disable_table @source_schema = N'dbo', @source_name = N'battery', @capture_instance = N'dbo_battery_data';
-- exec sys.sp_cdc_disable_table @source_schema = N'dbo', @source_name = N'motor', @capture_instance = N'dbo_motor_data';

-- Grant read only access of CDC tables
grant select on cdc.dbo_battery_data_CT to airbyte_user;
grant select on cdc.dbo_motor_data_CT to airbyte_user;

grant select on cdc.captured_columns to airbyte_user;
grant select on cdc.change_tables to airbyte_user;

-- Set retention period to 14400 minutes (10 days)
exec sp_cdc_change_job @job_type='cleanup', @retention = 14400;
-- Check: exec sys.sp_cdc_help_jobs;

-- Restart job
exec sys.sp_cdc_stop_job @job_type = 'cleanup';
exec sys.sp_cdc_start_job @job_type = 'cleanup';
-- Check: exec xp_servicecontrol 'QueryState', N'SQLServerAGENT';
