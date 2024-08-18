-- Production
use role USERADMIN;
create user if not exists PRODUCTION_TERRAFORM_USER RSA_PUBLIC_KEY='MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAyccSv7PVk/SFUPtOGMi6
i8ffEuv0Jp6I3/FQRoVS0Ux+eI9ec/zY0v9tdeqyNkDIsywBV82t7WLnirrgxM+Z
nS8TPKT7/0sd+CCpdVFxSGOK+mdro+krPp3jH+E099Ix4b7JVlcUXRWDwwCsEMY3
nEBn/r4dR/5qEiAQaOxeCLay3A5/ZIkEySxbqhi56hPC++4C8PMv7aEpJDJgPPNK
x7mDdtz0P0cGR/ybNcwzXITd67YGSlJrwndCVNUdl2Aw6JyB4UIIEI/cnxMVD8kr
pHu85qkb3L7CWKjjVcqRAECePpMC1Uo9JBgScLwLNgnAVJkAsXOdCv/M1zXE0tja
9/IxC8mt8e66bWxiPddhpfJXPfW11FGVwSW1R2ttrgW6JteDfJQAruDDfEIA5pes
LN9cCbpDCPGHMCAcq7u/+d6gjk+lQ1KpbSPdq+09kbnP9/Jp76gR3o7D6A3XVyV1
WcmkO6EB+p2id22/1rQEyQ6M5j7o2donHGG0R3DCAePyqrd1jP3flJQZkyLg5flw
R29bQ9b4z3XitfYdyk1mAm5LSQbAfGGeBChZdrdDbGb8USy3ZdwsizOJTSquQfgZ
wIROZuRix8TKblOxdxeGhOjNDOnJglsuEJjA//jYTyKdbjj5oOvi9jhaaWyFrYP+
llsCkh04J8wVAWFkG18KKJ8CAwEAAQ==' default_role=PUBLIC must_change_password=false;

use role SECURITYADMIN;
create role if not exists PRODUCTION_TERRAFORM_ROLE;
grant role PRODUCTION_TERRAFORM_ROLE to user PRODUCTION_TERRAFORM_USER;
grant create role on account to role PRODUCTION_TERRAFORM_ROLE;
grant create user on account to role PRODUCTION_TERRAFORM_ROLE;
grant manage grants on account to role PRODUCTION_TERRAFORM_ROLE;

use role SYSADMIN;
grant create database on account to role PRODUCTION_TERRAFORM_ROLE;
grant create warehouse on account to role PRODUCTION_TERRAFORM_ROLE;

grant role PRODUCTION_TERRAFORM_ROLE to role SECURITYADMIN;
grant role PRODUCTION_TERRAFORM_ROLE to role SYSADMIN;
