alter table orders add column status varchar(20) not null default 'new';
create index idx_orders_status on orders(status);
