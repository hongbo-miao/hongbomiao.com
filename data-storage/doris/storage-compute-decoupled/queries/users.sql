-- Binds each app to its own compute group via the native default_compute_group user property
-- (verified present in UserProperty.java on branch-4.1), not backend location tags.
create user if not exists 'application_a_user' identified by 'passw0rd';
grant select_priv on paimon_catalog.*.* to 'application_a_user';
grant usage_priv on compute group 'application_a' to 'application_a_user';
set property for 'application_a_user' 'default_compute_group' = 'application_a';

create user if not exists 'application_b_user' identified by 'passw0rd';
grant select_priv on paimon_catalog.*.* to 'application_b_user';
grant usage_priv on compute group 'application_b' to 'application_b_user';
set property for 'application_b_user' 'default_compute_group' = 'application_b';

show property for 'application_a_user';
show property for 'application_b_user';
