-- access_controller_type = ranger-doris disables Doris authorization entirely, so the grants that
-- used to live here are now Ranger policies (opentofu/policies/). default_compute_group stays: it
-- is a routing preference rather than a privilege, and Ranger has no equivalent.
create user if not exists 'application_a_user' identified by 'passw0rd';
set property for 'application_a_user' 'default_compute_group' = 'application_a';

create user if not exists 'application_b_user' identified by 'passw0rd';
set property for 'application_b_user' 'default_compute_group' = 'application_b';

show property for 'application_a_user';
show property for 'application_b_user';
