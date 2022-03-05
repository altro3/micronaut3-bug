package com.micronaut.bug.model;

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
    private MyStatus status;
}
