CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users
(
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email      TEXT UNIQUE NOT NULL,
    password   TEXT        NOT NULL,
    firstname  TEXT        NOT NULL,
    lastname   TEXT        NOT NULL,
    created_on TIMESTAMP        DEFAULT now(),
    last_login TIMESTAMP
);
