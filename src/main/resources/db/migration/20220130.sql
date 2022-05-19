create table if not exists myentity
(
    id serial primary key,
    field varchar(255),
    status char(1),
    json_field json,
    json_field_bin jsonb
);
