-- Check jobs
exec sys.sp_cdc_help_jobs;

exec sys.sp_cdc_stop_job @job_type = 'cleanup';
exec sys.sp_cdc_start_job @job_type = 'cleanup';

-- Check change data capture
exec sys.sp_cdc_help_change_data_capture
    @source_schema = N'dbo',
    @source_name = N'my_table';
