create table if not exists tweet_hashtag
(
    tweet_id text not null,
    hashtag_id uuid not null,
    constraint fk_space foreign key (hashtag_id) references hashtag (id)
);
