package com.micronaut.bug.dao;

import com.micronaut.bug.model.MyEntity;
import jakarta.inject.Singleton;

import java.util.List;

/*
@JdbcRepository(dialect = Dialect.POSTGRES)
public interface MyEntityDao extends CrudRepository<MyEntity, Integer> {
}
*/

@Singleton
public class MyEntityDao {

    public Iterable<MyEntity> findAll() {
        return List.of();
    }

    public MyEntity save(MyEntity myEntity) {
        return myEntity;
    }
}
