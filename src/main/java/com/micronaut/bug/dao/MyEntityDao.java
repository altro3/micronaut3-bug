package com.micronaut.bug.dao;

import com.micronaut.bug.model.MyEntity;
//import io.micronaut.data.jdbc.annotation.JdbcRepository;
//import io.micronaut.data.model.query.builder.sql.Dialect;
//import io.micronaut.data.repository.CrudRepository;
import io.micronaut.data.jdbc.annotation.JdbcRepository;
import io.micronaut.data.model.query.builder.sql.Dialect;
import io.micronaut.data.repository.CrudRepository;
import jakarta.inject.Singleton;

import java.util.List;

@JdbcRepository(dialect = Dialect.POSTGRES)
public interface MyEntityDao extends CrudRepository<MyEntity, Integer> {
}

/*
@Singleton
public class MyEntityDao{

    public Iterable<MyEntity> findAll() {
        return List.of();
    }
}
*/
