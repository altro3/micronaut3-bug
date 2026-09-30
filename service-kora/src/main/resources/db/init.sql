SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'service_kora'
  AND pid <> pg_backend_pid();

drop database if exists service_kora;

DROP ROLE IF EXISTS service_kora;

CREATE ROLE service_kora WITH LOGIN
    ENCRYPTED PASSWORD '1'
    SUPERUSER INHERIT CREATEDB CREATEROLE REPLICATION;

CREATE DATABASE service_kora
    WITH
    OWNER = service_kora
    ENCODING = 'UTF8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

\c service_kora

CREATE SCHEMA custom AUTHORIZATION service_kora;

GRANT ALL PRIVILEGES ON SCHEMA custom TO service_kora;
