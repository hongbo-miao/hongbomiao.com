conn = database('archer-snowflake', 'production_tracker_db_public_readonly', 'xxx')

fieldNames = {'record_metadata', 'record_content'};

columnNames = strjoin(fieldNames, ', ');
sqlQuery = [
            'select ', columnNames, ' ', ...
            'from production_tracker_db.public.analytic_events'
           ];
data = fetch(conn, sqlQuery);

close(conn);

clear columnNames;
clear conn;
clear fieldNames;
clear sqlQuery;
