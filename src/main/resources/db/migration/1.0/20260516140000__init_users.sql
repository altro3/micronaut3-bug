create table users
(
    id serial primary key,
    username varchar(50) not null unique
);
