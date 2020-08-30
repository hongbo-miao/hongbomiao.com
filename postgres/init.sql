CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users
(
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email      TEXT UNIQUE NOT NULL,
    password   TEXT        NOT NULL,
    first_name TEXT        NOT NULL,
    last_name  TEXT        NOT NULL,
    bio        TEXT,
    created_on TIMESTAMP        DEFAULT now(),
    last_login TIMESTAMP
);
