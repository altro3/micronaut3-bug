SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'micronaut3bug'
  AND pid <> pg_backend_pid();

drop database if exists micronaut3bug;

DROP ROLE IF EXISTS micronaut3bug;

CREATE ROLE micronaut3bug WITH LOGIN
    ENCRYPTED PASSWORD '1'
    SUPERUSER INHERIT CREATEDB CREATEROLE REPLICATION;

CREATE DATABASE micronaut3bug
    WITH
    OWNER = micronaut3bug
    ENCODING = 'UTF8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;
