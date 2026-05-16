drop index if exists idx_orders_status;
alter table orders drop column if exists status;
