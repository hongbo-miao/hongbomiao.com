create table if not exists tweets
(
  timestamp timestamp without time zone not null,
  id        text                        not null,
  id_str    text                        not null,
  text      text                        not null,
  lang      text                        not null
);

select create_hypertable('tweets', 'timestamp');
