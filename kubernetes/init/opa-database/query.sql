select data
from opa
where org = 'hm';

select data -> 'user_roles' as user_roles
from opa
where org = 'hm';

select data #> '{user_roles, 0x1}' as user_role
from opa
where org = 'hm';
