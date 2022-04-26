show catalogs;
show schemas in tpcds;
show tables from tpch.sf1;

select count(name) from tpch.sf1.nation;

use tpch.sf1;
select count(name) from nation;

select nationkey, name, regionkey from tpch.sf1.nation limit 5;
