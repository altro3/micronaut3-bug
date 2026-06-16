SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'orchestrator_spring'
  AND pid <> pg_backend_pid();

drop database if exists orchestrator_spring;

DROP ROLE IF EXISTS orchestrator_spring;

CREATE ROLE orchestrator_spring WITH LOGIN
    ENCRYPTED PASSWORD '1'
    SUPERUSER INHERIT CREATEDB CREATEROLE REPLICATION;

CREATE DATABASE orchestrator_spring
    WITH
    OWNER = orchestrator_spring
    ENCODING = 'UTF8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

\c orchestrator_spring

CREATE SCHEMA target AUTHORIZATION orchestrator_spring;

GRANT ALL PRIVILEGES ON SCHEMA target TO orchestrator_spring;
