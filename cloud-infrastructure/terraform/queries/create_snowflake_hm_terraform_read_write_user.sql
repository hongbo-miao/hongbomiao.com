-- Development
use role USERADMIN;
create user if not exists HM_DEVELOPMENT_TERRAFORM_READ_WRITE_USER RSA_PUBLIC_KEY='MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAvHV54FA7Ww0HmpUYfR+U
OnvrJG050pvZX128FSXCEJ7PPJuug3iq7FdrO7Ge2KYYOas/v0blw1eWHyQt+12Q
LIsoeEerO1F/w78OGnsXWfvkP9Tg1advhwPpDZZ3TG/AljPpvBX/ZIkUOXQq0DsY
rY/f+Dx+DhR1FyPWqbrKx511g4DSGVrmk0uL8ib+Q/JU3LSvOL7tP12pBQJgXMSh
B5F8+vJeeyeGrc5QJeNclzLv91s/9oZ5SVM9JMcFdm6IhO4v0w/V8PrL7OCA2JoR
sIkMxF7Nu6gtEMDjyXCT8v2dXlWfmxqq8KO5kpMXUN8jeGURWPR5K1GabjVCuBK3
/Cuze2DkVQ32AmtdYxTy+PZDRJ6UUY2rvvxMaqSWPut8Cz8jU/760KMRiZs5tJ1p
QJ3ONdfkPICFXDfiUq3MIjTbmIj/QrUZIk1aHoV7pOkx3I+BmkLr0nfHgQ5By6gK
iWyHRZBeCXbnc3UbUJ1u6Ombme9RazYq/5WINSxZNtkdtG5EYNT6pfVxhRsXZsv3
Dm29F2gzgIOu9eTGWZjfkyty2Ly9c07RUhK+Z9l4xn2eOCT1rlsjhNKD26H14BOW
KxSvNcGk8G2hawdT2AZN44PTrrogVfubQUswJMquXE7PbUdsjGM27ht8XR9X8ra8
5wSTCvlrmBZEQU5z61s/eKUCAwEAAQ==' default_role=PUBLIC must_change_password=false;

use role SECURITYADMIN;
create role if not exists HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE;
grant role HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE to user HM_DEVELOPMENT_TERRAFORM_READ_WRITE_USER;
grant create role on account to role HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE;
grant create user on account to role HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE;
grant manage grants on account to role HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE;

use role SYSADMIN;
grant create database on account to role HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE;
grant create warehouse on account to role HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE;

grant role HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE to role SECURITYADMIN;
grant role HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE to role SYSADMIN;

-- Production
use role USERADMIN;
create user if not exists HM_PRODUCTION_TERRAFORM_READ_WRITE_USER RSA_PUBLIC_KEY='MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAyccSv7PVk/SFUPtOGMi6
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
create role if not exists HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE;
grant role HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE to user HM_PRODUCTION_TERRAFORM_READ_WRITE_USER;
grant create role on account to role HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE;
grant create user on account to role HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE;
grant manage grants on account to role HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE;

use role SYSADMIN;
grant create database on account to role HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE;
grant create warehouse on account to role HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE;

grant role HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE to role SECURITYADMIN;
grant role HM_PRODUCTION_TERRAFORM_READ_WRITE_ROLE to role SYSADMIN;
