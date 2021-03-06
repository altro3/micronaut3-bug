package com.micronaut.bug.model;

import io.micronaut.core.annotation.Introspected;
import lombok.*;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@Builder
@ToString
//@MappedEntity("myentity")
public class MyEntity {

    //    @Id
//    @GeneratedValue(value = GeneratedValue.Type.SEQUENCE, ref = "myentity_id_seq")
    private int id;
    private String field;
    //    @TypeDef(type = DataType.CHARACTER, converter = MyStatusConverter.class)
    private MyStatus status;
    //    @TypeDef(type = DataType.JSON)
    private JsonField jsonField;
    //    @TypeDef(type = DataType.JSON)
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
