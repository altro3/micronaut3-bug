package com.micronaut.bug.model;

import io.micronaut.core.annotation.Introspected;
import io.micronaut.data.annotation.GeneratedValue;
import io.micronaut.data.annotation.Id;
import io.micronaut.data.annotation.MappedEntity;
import io.micronaut.data.annotation.TypeDef;
import io.micronaut.data.model.DataType;
import lombok.*;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@Builder
@ToString
@MappedEntity("myentity")
public class MyEntity {

    @Id
    @GeneratedValue(value = GeneratedValue.Type.SEQUENCE, ref = "myentity_id_seq")
    private int id;
    private String field;
    @TypeDef(type = DataType.CHARACTER, converter = MyStatusConverter.class)
    private MyStatus status;
    @TypeDef(type = DataType.JSON)
    private JsonField jsonField;
    @TypeDef(type = DataType.JSON)
    private JsonField jsonFieldBin;

    //@Serdeable
    @Introspected
    @NoArgsConstructor
    @AllArgsConstructor
    @Getter
    @Setter
    @Builder
    @ToString
    public static class JsonField {

        private int id;
        private String text;
        private Long number;
    }
}
