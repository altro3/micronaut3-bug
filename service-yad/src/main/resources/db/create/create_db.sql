SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'yad'
  AND pid <> pg_backend_pid();

drop database if exists yad;

DROP ROLE IF EXISTS yad;

CREATE ROLE yad WITH LOGIN
    ENCRYPTED PASSWORD '1'
    SUPERUSER INHERIT CREATEDB CREATEROLE REPLICATION;

CREATE DATABASE yad
    WITH
    OWNER = yad
    ENCODING = 'UTF8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

\c yad

CREATE SCHEMA target AUTHORIZATION yad;

GRANT ALL PRIVILEGES ON SCHEMA target TO yad;
