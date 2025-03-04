create user odoo_user with encrypted password 'xxx';

-- Odoo requires that odoo_user owns odoo_db
grant odoo_user to postgres;
create database odoo_db with owner odoo_user;
