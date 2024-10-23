-- Create user
create login hongbo_miao with password = 'xxx';
create login hongbo_miao with password = 'xxx', default_database = master;
create user hongbo_miao for login hongbo_miao;

-- Drop user
drop user hongbo_miao;
drop login hongbo_miao;

-- Check login failed log
exec xp_readerrorlog 0, 1, N'Login failed';

-- Unlock user
alter login hongbo_miao with password = 'xxx' unlock;

-- List server principals
select * from sys.server_principals;
