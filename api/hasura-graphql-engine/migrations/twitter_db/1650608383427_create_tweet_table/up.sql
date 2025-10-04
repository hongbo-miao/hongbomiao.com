create table if not exists tweet
(
    timestamp timestamp without time zone not null,
    id text not null,
    twitter_user_id text not null,
    text text not null,
    lang text not null
);

select create_hypertable('tweet', 'timestamp');
