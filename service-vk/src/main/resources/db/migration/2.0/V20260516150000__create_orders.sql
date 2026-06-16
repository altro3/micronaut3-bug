create table orders
(
    id serial primary key,
    user_id int not null references users (id) on delete cascade,
    amount numeric(12, 2) not null
);
